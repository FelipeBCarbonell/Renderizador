#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from utils import *

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    # Pilha de transformações do modelo. O topo (índice -1) é sempre a
    # matriz de transformação acumulada (mundo <- objeto) válida no ponto
    # atual do grafo de cena. Começa com a identidade.
    transform_stack = [np.identity(4)]

    # Matriz de visualização (view) e de projeção perspectiva, calculadas
    # em viewpoint(). Começam como identidade até que um Viewpoint seja lido.
    view_matrix = np.identity(4)
    perspective_matrix = np.identity(4)

    # Anti-aliasing por superamostragem (lado do bloco de amostras por pixel;
    # 2 = 2x2 amostras). Use 1 para desligar.
    supersampling = 2

    # Buffers superamostrados do quadro atual (criados em begin_frame)
    _ss_color = None    # cor (0-255) de cada amostra
    _ss_depth = None    # z-buffer de cada amostra
    _ss_covered = None  # amostras desenhadas por triângulos neste quadro

    _mip_cache = {}     # pirâmides de mipmap por textura

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    # ------------------------------------------------------------------
    # Utilitários de matrizes (transformações geométricas em coordenadas
    # homogêneas, vetores coluna: p' = M @ p)
    # ------------------------------------------------------------------

    @staticmethod
    def _translation_matrix(t):
        """Monta a matriz 4x4 de translação a partir de [x, y, z]."""
        return np.array([
            [1, 0, 0, t[0]],
            [0, 1, 0, t[1]],
            [0, 0, 1, t[2]],
            [0, 0, 0, 1]
        ], dtype=float)

    @staticmethod
    def _scale_matrix(s):
        """Monta a matriz 4x4 de escala a partir de [sx, sy, sz]."""
        return np.array([
            [s[0], 0, 0, 0],
            [0, s[1], 0, 0],
            [0, 0, s[2], 0],
            [0, 0, 0, 1]
        ], dtype=float)

    @staticmethod
    def _rotation_matrix(r):
        """Monta a matriz 4x4 de rotação a partir de [x, y, z, angulo]
        (rotação ao redor do eixo (x, y, z) por 'angulo' radianos, regra
        da mão direita), usando a fórmula de Rodrigues."""
        x, y, z, angle = r
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-8:
            return np.identity(4)
        x, y, z = x / norm, y / norm, z / norm
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        return np.array([
            [t * x * x + c,     t * x * y - s * z, t * x * z + s * y, 0],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x, 0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c,     0],
            [0, 0, 0, 1]
        ], dtype=float)

    @staticmethod
    def _material(colors):
        """Extrai (cor emissiva em 0-255, transparência) do dicionário de cores."""
        emissive = colors.get("emissiveColor", [1, 1, 1])
        rgb = np.array([c * 255 for c in emissive], dtype=float)
        return rgb, float(colors.get("transparency", 0.0))

    @staticmethod
    def _get_point3d(coord, idx):
        """Extrai o vértice (x, y, z) de índice `idx` de uma lista plana de
        coordenadas [x0, y0, z0, x1, y1, z1, ...]."""
        return coord[idx * 3], coord[idx * 3 + 1], coord[idx * 3 + 2]

    # ------------------------------------------------------------------
    # Quadro: buffers superamostrados (anti-aliasing), z-buffer e composição
    # ------------------------------------------------------------------

    @staticmethod
    def begin_frame():
        """Prepara os buffers superamostrados (cor, profundidade e cobertura).
        Deve ser chamada no início de cada quadro (Renderizador.pre)."""
        s = GL.supersampling
        h, w = GL.height * s, GL.width * s
        GL._ss_color = np.empty((h, w, 3), dtype=float)
        GL._ss_color[:] = np.array(gpu.GPU.clear_color_val, dtype=float)[:3]
        GL._ss_depth = np.full((h, w), np.inf)    # z-buffer (NDC, menor = mais perto)
        GL._ss_covered = np.zeros((h, w), dtype=bool)
        GL.transform_stack = [np.identity(4)]

    @staticmethod
    def end_frame():
        """Reduz o buffer superamostrado (média s x s de cada pixel) e grava
        no framebuffer. Amostras não cobertas por triângulos mantêm o que já
        estava no framebuffer (por exemplo, desenhos 2D de pontos e linhas).
        Deve ser chamada no fim de cada quadro (Renderizador.pos)."""
        if GL._ss_covered is None or not GL._ss_covered.any():
            return
        s = GL.supersampling
        fb = gpu.GPU.get_frame_buffer()
        base = np.repeat(np.repeat(fb[:, :, :3], s, axis=0), s, axis=1).astype(float)
        final = np.where(GL._ss_covered[..., None], GL._ss_color, base)
        final = final.reshape(GL.height, s, GL.width, s, 3).mean(axis=(1, 3))
        fb[:, :, :3] = np.clip(np.rint(final), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Texturas (mipmap)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mipmaps(name):
        """Carrega a textura e monta (com cache) sua pirâmide de mipmaps: cada
        nível é a média 2x2 do anterior. Retorna lista de arrays [x][y][rgb]
        (mesma orientação de gpu.GPU.load_texture)."""
        key = (gpu.GPU.path, name)
        if key not in GL._mip_cache:
            img = gpu.GPU.load_texture(name).astype(float)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            levels = [img[:, :, :3]]

            def half(a, axis):
                n = a.shape[axis]
                if n == 1:
                    return a
                n -= n % 2
                even = np.take(a, range(0, n, 2), axis=axis)
                odd = np.take(a, range(1, n, 2), axis=axis)
                return (even + odd) / 2

            while min(levels[-1].shape[:2]) > 1:
                levels.append(half(half(levels[-1], 0), 1))
            GL._mip_cache[key] = levels
        return GL._mip_cache[key]

    @staticmethod
    def _sample_mipmaps(mipmaps, u, v, dudx, dvx, dudy, dvy):
        """Amostra a textura em (u, v) escolhendo o nível de mipmap pela
        razão texels/pixel (derivadas de u, v em relação à tela)."""
        w0, h0 = mipmaps[0].shape[:2]
        rho = np.maximum(np.hypot(dudx * w0, dvx * h0), np.hypot(dudy * w0, dvy * h0))
        lod = np.log2(np.maximum(rho, 1e-8))
        level = np.clip(np.floor(lod + 0.5).astype(int), 0, len(mipmaps) - 1)

        out = np.empty((u.size, 3))
        for lv in np.unique(level):
            sel = level == lv
            tex = mipmaps[lv]
            wl, hl = tex.shape[:2]
            col = np.clip(np.floor(u[sel] * wl).astype(int), 0, wl - 1)
            row = np.clip(np.floor((1.0 - v[sel]) * hl).astype(int), 0, hl - 1)  # t=0 é a base da imagem
            out[sel] = tex[col, row]
        return out

    # ------------------------------------------------------------------
    # Rasterização
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_triangle(pts, zs, ws, color=None, vcolors=None, uvs=None,
                       mipmaps=None, transparency=0.0, depth_test=True):
        """Rasteriza um triângulo no buffer superamostrado.

        pts : 3 vértices (x, y) em pixels de tela (não superamostrados)
        zs  : profundidade NDC de cada vértice (interpolada linearmente na tela)
        ws  : w de clip de cada vértice (usado na correção de perspectiva)
        color / vcolors / uvs+mipmaps : cor sólida (0-255), cores por vértice
            (0-255) ou coordenadas de textura por vértice.
        """
        s = GL.supersampling
        sx = [p[0] * s for p in pts]
        sy = [p[1] * s for p in pts]
        x0, x1, x2 = sx
        y0, y1, y2 = sy
        area = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)
        if abs(area) < 1e-12:
            return

        xmin = max(0, int(math.floor(min(sx))))
        xmax = min(GL.width * s - 1, int(math.ceil(max(sx))))
        ymin = max(0, int(math.floor(min(sy))))
        ymax = min(GL.height * s - 1, int(math.ceil(max(sy))))
        if xmin > xmax or ymin > ymax:
            return
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))

        def bary(px, py):
            """Coordenadas baricêntricas (em tela) dos pontos (px, py)."""
            l0 = ((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)) / area
            l1 = ((px - x2) * (y0 - y2) - (py - y2) * (x0 - x2)) / area
            return l0, l1, 1.0 - l0 - l1

        # Cobertura: amostra no centro do pixel superamostrado. A pequena
        # tolerância evita "costuras" em arestas compartilhadas por erro de
        # arredondamento (amostras exatamente na aresta caem em ambos).
        eps = -1e-9
        b0, b1, b2 = bary(xs + 0.5, ys + 0.5)
        mask = (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
        if not mask.any():
            return
        ix, iy = xs[mask], ys[mask]
        z = b0[mask] * zs[0] + b1[mask] * zs[1] + b2[mask] * zs[2]

        # Recorte near/far e teste de profundidade (z-buffer)
        keep = (z >= -1.0) & (z <= 1.0)
        if depth_test:
            keep &= z < GL._ss_depth[iy, ix]
        if not keep.any():
            return
        ix, iy, z = ix[keep], iy[keep], z[keep]
        fx, fy = ix + 0.5, iy + 0.5

        def persp(px, py):
            """Pesos com correção de perspectiva: λi/wi normalizados."""
            l0, l1, l2 = bary(px, py)
            a0, a1, a2 = l0 / ws[0], l1 / ws[1], l2 / ws[2]
            tot = a0 + a1 + a2
            tot = np.where(np.abs(tot) < 1e-12, 1e-12, tot)
            return a0 / tot, a1 / tot, a2 / tot

        if mipmaps is not None:
            def uv_at(px, py):
                q = persp(px, py)
                return (q[0] * uvs[0][0] + q[1] * uvs[1][0] + q[2] * uvs[2][0],
                        q[0] * uvs[0][1] + q[1] * uvs[1][1] + q[2] * uvs[2][1])
            u, v = uv_at(fx, fy)
            ux, vx = uv_at(fx + 1, fy)
            uy, vy = uv_at(fx, fy + 1)
            col = GL._sample_mipmaps(mipmaps, u, v, ux - u, vx - v, uy - u, vy - v)
        elif vcolors is not None:
            p = persp(fx, fy)
            vc = np.asarray(vcolors, dtype=float)
            col = p[0][:, None] * vc[0] + p[1][:, None] * vc[1] + p[2][:, None] * vc[2]
        else:
            col = np.broadcast_to(np.asarray(color, dtype=float), (ix.size, 3))

        # Composição de transparência: a cor de trás pesa `transparency`
        if transparency > 0:
            col = col * (1.0 - transparency) + GL._ss_color[iy, ix] * transparency

        GL._ss_color[iy, ix] = col
        GL._ss_covered[iy, ix] = True
        if depth_test and transparency <= 0:
            GL._ss_depth[iy, ix] = z  # transparentes testam, mas não escrevem profundidade

    @staticmethod
    def _draw_triangle_3d(p0, p1, p2, mvp, rgb, transparency=0.0,
                          vcolors=None, uvs=None, mipmaps=None):
        """Projeta três vértices (x, y, z) do espaço do objeto com a matriz
        `mvp`, faz a divisão perspectiva, leva à tela e rasteriza."""
        pts, zs, ws = [], [], []
        for (x, y, z) in (p0, p1, p2):
            clip = mvp @ np.array([x, y, z, 1.0])
            w = clip[3]
            if w <= 1e-8:  # atrás da câmera (sem recorte de triângulos parciais)
                return
            ndc = clip[:3] / w
            pts.append(((ndc[0] + 1) * GL.width / 2, (1 - ndc[1]) * GL.height / 2))
            zs.append(ndc[2])
            ws.append(w)
        GL._fill_triangle(pts, zs, ws, color=rgb, vcolors=vcolors, uvs=uvs,
                          mipmaps=mipmaps, transparency=transparency)
    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        emissive = colors.get("emissiveColor", [1, 1, 1])
        rgb = [int(round(c * 255)) for c in emissive]

        for i in range(0, len(point), 2):
            x = int(round(point[i]))
            y = int(round(point[i + 1]))
            if 0 <= x < GL.width and 0 <= y < GL.height:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        emissive = colors.get("emissiveColor", [1, 1, 1])
        rgb = [int(round(c * 255)) for c in emissive]

        pts = [(lineSegments[i], lineSegments[i + 1])
            for i in range(0, len(lineSegments), 2)]

        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            draw_line(x0, y0, x1, y1, rgb)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        emissive = colors.get("emissiveColor", [1, 1, 1])
        rgb = [int(round(c * 255)) for c in emissive]

        # Agrupa de 6 em 6 valores (3 vértices x, y por triângulo). Passa pelo
        # buffer superamostrado (anti-aliasing), sem teste de profundidade.
        for t in range(0, len(vertices) - 5, 6):
            pts = [(vertices[t], vertices[t + 1]),
                   (vertices[t + 2], vertices[t + 3]),
                   (vertices[t + 4], vertices[t + 5])]
            GL._fill_triangle(pts, [0, 0, 0], [1, 1, 1], color=rgb, depth_test=False)


    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        rgb, transp = GL._material(colors)

        # Matriz Model-View-Projection: leva pontos do espaço do objeto
        # (local) até o espaço de recorte (clip space). A divisão perspectiva
        # e o mapeamento para tela são feitos por vértice em _draw_triangle_3d.
        mvp = GL.perspective_matrix @ GL.view_matrix @ GL.transform_stack[-1]

        # Agrupa de 9 em 9 valores (3 vértices x, y, z por triângulo)
        for t in range(0, len(point) - 8, 9):
            p0 = (point[t], point[t + 1], point[t + 2])
            p1 = (point[t + 3], point[t + 4], point[t + 5])
            p2 = (point[t + 6], point[t + 7], point[t + 8])
            GL._draw_triangle_3d(p0, p1, p2, mvp, rgb, transp)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # Matriz que leva a câmera da origem/orientação padrão até sua pose
        # no mundo (posição + orientação). A view matrix é o inverso disso,
        # pois transformamos o mundo para o espaço da câmera.
        camera_matrix = GL._translation_matrix(position) @ GL._rotation_matrix(orientation)
        GL.view_matrix = np.linalg.inv(camera_matrix)

        # fieldOfView do X3D é o MENOR entre o campo de visão horizontal e
        # vertical. Em telas largas (aspect >= 1) ele já é o vertical (fovy);
        # em telas altas é o horizontal e é preciso converter para fovy:
        aspect = GL.width / GL.height
        fovy = fieldOfView
        if aspect < 1:
            fovy = 2 * math.atan(math.tan(fieldOfView / 2) / aspect)

        top = GL.near * math.tan(fovy / 2)
        right = top * aspect

        GL.perspective_matrix = np.array([
            [GL.near / right, 0, 0, 0],
            [0, GL.near / top, 0, 0],
            [0, 0, -(GL.far + GL.near) / (GL.far - GL.near),
             -2 * GL.far * GL.near / (GL.far - GL.near)],
            [0, 0, -1, 0]
        ], dtype=float)

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        T = GL._translation_matrix(translation) if translation else np.identity(4)
        R = GL._rotation_matrix(rotation) if rotation else np.identity(4)
        S = GL._scale_matrix(scale) if scale else np.identity(4)

        # Ordem padrão X3D: escala, depois rotação, depois translação
        # (aplicada da direita para a esquerda sobre o ponto).
        local_matrix = T @ R @ S

        parent_matrix = GL.transform_stack[-1]
        GL.transform_stack.append(parent_matrix @ local_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if len(GL.transform_stack) > 1:
            GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        rgb, transp = GL._material(colors)
        mvp = GL.perspective_matrix @ GL.view_matrix @ GL.transform_stack[-1]

        offset = 0
        for count in stripCount:
            for j in range(count - 2):
                p0 = GL._get_point3d(point, offset + j)
                p1 = GL._get_point3d(point, offset + j + 1)
                p2 = GL._get_point3d(point, offset + j + 2)
                # Tiras de triângulo alternam a orientação dos vértices a
                # cada triângulo para manter a face consistente (sentido
                # horário/anti-horário) ao longo da tira.
                if j % 2 == 0:
                    GL._draw_triangle_3d(p0, p1, p2, mvp, rgb, transp)
                else:
                    GL._draw_triangle_3d(p1, p0, p2, mvp, rgb, transp)
            offset += count

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        rgb, transp = GL._material(colors)
        mvp = GL.perspective_matrix @ GL.view_matrix @ GL.transform_stack[-1]

        strip = []
        for idx in index:
            if idx == -1:
                # -1 marca o fim de uma tira; a próxima tira recomeça do zero
                strip = []
                continue
            strip.append(idx)
            if len(strip) >= 3:
                j = len(strip) - 3
                p0 = GL._get_point3d(point, strip[j])
                p1 = GL._get_point3d(point, strip[j + 1])
                p2 = GL._get_point3d(point, strip[j + 2])
                # Alterna a orientação a cada triângulo para manter a face consistente
                if j % 2 == 0:
                    GL._draw_triangle_3d(p0, p1, p2, mvp, rgb, transp)
                else:
                    GL._draw_triangle_3d(p1, p0, p2, mvp, rgb, transp)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        rgb, transp = GL._material(colors)
        mvp = GL.perspective_matrix @ GL.view_matrix @ GL.transform_stack[-1]

        # Textura: só usada se houver imagem e coordenadas de textura
        mipmaps = None
        if current_texture and texCoord:
            mipmaps = GL._get_mipmaps(current_texture[0])

        per_vertex_color = bool(color) and bool(colorPerVertex) and mipmaps is None
        color_idx = colorIndex if colorIndex else coordIndex
        tex_idx = texCoordIndex if texCoordIndex else coordIndex

        def draw_face(face, start, number):
            """Triangula a face em leque e desenha cada triângulo. `start` é a
            posição do primeiro vértice da face em coordIndex (as listas de
            índices de cor/textura são paralelas a ela) e `number` o número
            da face (usado para cor por face)."""
            if len(face) < 3:
                return

            face_rgb = rgb
            if color and not colorPerVertex and mipmaps is None:
                ci = colorIndex[number] if colorIndex else number
                face_rgb = np.array(color[ci * 3:ci * 3 + 3]) * 255

            verts = []
            for k, vi in enumerate(face):
                vc = uv = None
                if mipmaps is not None:
                    t = tex_idx[start + k]
                    uv = (texCoord[t * 2], texCoord[t * 2 + 1])
                elif per_vertex_color:
                    c = color_idx[start + k]
                    vc = np.array(color[c * 3:c * 3 + 3]) * 255
                verts.append((GL._get_point3d(coord, vi), vc, uv))

            for k in range(1, len(face) - 1):
                tri = (verts[0], verts[k], verts[k + 1])
                GL._draw_triangle_3d(
                    tri[0][0], tri[1][0], tri[2][0], mvp, face_rgb, transp,
                    vcolors=[t[1] for t in tri] if per_vertex_color else None,
                    uvs=[t[2] for t in tri] if mipmaps is not None else None,
                    mipmaps=mipmaps)

        face, start, number = [], 0, 0
        for i, idx in enumerate(coordIndex):
            if idx == -1:
                # -1 marca o fim de uma face
                draw_face(face, start, number)
                face, start, number = [], i + 1, number + 1
                continue
            face.append(idx)
        draw_face(face, start, number)  # segurança, caso a lista não termine com -1

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""