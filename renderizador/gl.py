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

    # Iluminação: luzes coletadas em cada quadro e dados da câmera
    _lights = []                # DirectionalLight/PointLight lidas neste quadro
    _headlight = False          # luz da câmera (NavigationInfo headlight)
    _eye = np.zeros(3)          # posição da câmera no mundo
    _cam_rot = np.identity(3)   # orientação da câmera (3x3)

    # Em tiras de triângulos as normais são suavizadas entre faces vizinhas
    # (vértices compartilhados). Use False para sombreamento facetado.
    smooth_strips = True

    _timers = {}                # instante de início de cada TimeSensor

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL._timers = {}

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
        """Reúne as propriedades do material. Cores em 0-1, exceto a emissiva
        (0-255). 'lit' é falso quando o Shape não tem nó Material: nesse caso
        não há iluminação e a cor vem da textura/cores por vértice (ou branco)."""
        lit = bool(colors.get("material", True))
        emissive = np.array(colors.get("emissiveColor", [1, 1, 1]), dtype=float)
        if not lit:
            emissive = np.ones(3)
        return {
            "lit": lit,
            "emissive": emissive * 255,
            "diffuse": np.array(colors.get("diffuseColor", [0.8, 0.8, 0.8]), dtype=float),
            "specular": np.array(colors.get("specularColor", [0, 0, 0]), dtype=float),
            "shininess": float(colors.get("shininess", 0.2)),
            "ambient": float(colors.get("ambientIntensity", 0.2)),
            "transparency": float(colors.get("transparency", 0.0)),
        }

    @staticmethod
    def _get_point3d(coord, idx):
        """Extrai o vértice (x, y, z) de índice `idx` de uma lista plana de
        coordenadas [x0, y0, z0, x1, y1, z1, ...]."""
        return coord[idx * 3], coord[idx * 3 + 1], coord[idx * 3 + 2]

    # ------------------------------------------------------------------
    # Iluminação (modelo do X3D: emissiva + ambiente + difusa + especular)
    # ------------------------------------------------------------------

    @staticmethod
    def _active_lights():
        """Luzes do quadro: as da cena mais a headlight (se ligada), que
        sempre aponta para onde a câmera olha (-Z da câmera)."""
        lights = list(GL._lights)
        if GL._headlight:
            lights.append({"type": "dir", "dir": GL._cam_rot @ np.array([0.0, 0.0, -1.0]),
                           "color": np.ones(3), "intensity": 1.0, "ambient": 0.0})
        return lights

    @staticmethod
    def _shade(P, N, base, mat):
        """Calcula a cor (0-255) de cada fragmento.

        P    : (n, 3) posições no mundo
        N    : (n, 3) normais no mundo
        base : (n, 3) ou (3,) cor difusa (0-1) — material, textura ou vértice
        mat  : dicionário de _material

        cor = emissiva + soma por luz de cor_da_luz * (ambiente + difusa + especular)
          ambiente  = ambientIntensity_luz * ambientIntensity_material * base
          difusa    = intensidade * base * (N . L)
          especular = intensidade * specularColor * (N . H) ^ (shininess * 128)
        """
        n = len(P)
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        V = GL._eye - P
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        # Iluminação de dois lados: vira a normal para o lado de quem olha
        N = np.where(((N * V).sum(axis=1) < 0)[:, None], -N, N)

        out = np.tile(mat["emissive"] / 255.0, (n, 1))
        for light in GL._active_lights():
            if light["type"] == "dir":
                L = -light["dir"]                    # direção PARA a luz
            else:
                L = light["loc"] - P
                L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)
            nl = np.clip((N * L).sum(axis=1), 0.0, None)
            H = L + V
            H = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
            nh = np.clip((N * H).sum(axis=1), 0.0, None)
            spec = np.where(nl > 0, nh ** (mat["shininess"] * 128.0), 0.0)
            term = light["ambient"] * mat["ambient"] * base \
                + light["intensity"] * (base * nl[:, None] + mat["specular"] * spec[:, None])
            out = out + light["color"] * term
        return np.clip(out * 255.0, 0, 255)

    @staticmethod
    def _vertex_normals(P, tris, model):
        """Normais suaves (no mundo) por vértice: soma das normais das faces
        que dividem o vértice (ponderada pela área) e normalização."""
        P = np.asarray(P, dtype=float)
        W = (model @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        t = np.asarray(tris)
        fn = np.cross(W[t[:, 1]] - W[t[:, 0]], W[t[:, 2]] - W[t[:, 0]])
        vn = np.zeros_like(W)
        for k in range(3):
            np.add.at(vn, t[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _draw_indexed_mesh(P, tris, mat, vnormals=None, smooth=False):
        """Desenha uma malha de triângulos indexada (vértices P, triângulos
        `tris`). Normais: analíticas (vnormals, no espaço do objeto), suaves
        (smooth) ou, se nenhuma das duas, uma por face (facetado)."""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        if len(tris) == 0:
            return
        model = GL.transform_stack[-1]
        mvp = GL.perspective_matrix @ GL.view_matrix @ model

        normals = None
        if mat["lit"]:
            if vnormals is not None:
                # normais se transformam pela inversa transposta do modelo
                nm = np.linalg.pinv(model[:3, :3]).T
                normals = (nm @ np.asarray(vnormals, dtype=float).T).T
            elif smooth:
                normals = GL._vertex_normals(P, tris, model)

        for (a, b, c) in tris:
            GL._draw_triangle_3d(P[a], P[b], P[c], mvp, mat, model,
                                 None if normals is None else normals[[a, b, c]])

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
        GL._lights = []

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
                       mipmaps=None, transparency=0.0, depth_test=True, shade=None):
        """Rasteriza um triângulo no buffer superamostrado.

        pts : 3 vértices (x, y) em pixels de tela (não superamostrados)
        zs  : profundidade NDC de cada vértice (interpolada linearmente na tela)
        ws  : w de clip de cada vértice (usado na correção de perspectiva)
        color / vcolors / uvs+mipmaps : cor sólida (0-255), cores por vértice
            (0-255) ou coordenadas de textura por vértice.
        shade : None (sem luz) ou (W, N, mat): posições (3x3) e normais (3x3)
            dos vértices no mundo e o material; ativa o cálculo de iluminação
            por fragmento (a cor de vértice/textura passa a ser a cor difusa).
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

        weights = None
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
            weights = persp(fx, fy)
            vc = np.asarray(vcolors, dtype=float)
            col = (weights[0][:, None] * vc[0] + weights[1][:, None] * vc[1]
                   + weights[2][:, None] * vc[2])
        else:
            col = None  # cor sólida: definida abaixo

        if shade is not None:
            # Iluminação por fragmento: interpola posição e normal no mundo
            W, Nv, mat = shade
            if weights is None:
                weights = persp(fx, fy)
            P = (weights[0][:, None] * W[0] + weights[1][:, None] * W[1]
                 + weights[2][:, None] * W[2])
            N = (weights[0][:, None] * Nv[0] + weights[1][:, None] * Nv[1]
                 + weights[2][:, None] * Nv[2])
            base = col / 255.0 if col is not None else mat["diffuse"]
            col = GL._shade(P, N, base, mat)
        elif col is None:
            col = np.broadcast_to(np.asarray(color, dtype=float), (ix.size, 3))

        # Composição de transparência: a cor de trás pesa `transparency`
        if transparency > 0:
            col = col * (1.0 - transparency) + GL._ss_color[iy, ix] * transparency

        GL._ss_color[iy, ix] = col
        GL._ss_covered[iy, ix] = True
        if depth_test and transparency <= 0:
            GL._ss_depth[iy, ix] = z  # transparentes testam, mas não escrevem profundidade

    @staticmethod
    def _draw_triangle_3d(p0, p1, p2, mvp, mat, model=None, normals=None,
                          vcolors=None, uvs=None, mipmaps=None):
        """Projeta três vértices (x, y, z) do espaço do objeto com a matriz
        `mvp`, faz a divisão perspectiva, leva à tela e rasteriza. Se o material
        for iluminado, também leva os vértices ao mundo (com `model`) e calcula
        a normal da face (ou usa `normals`, uma por vértice, já no mundo)."""
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

        shade = None
        if mat["lit"]:
            corners = np.array([[*p0, 1.0], [*p1, 1.0], [*p2, 1.0]])
            W = (model @ corners.T).T[:, :3]
            if normals is None:
                n = np.cross(W[1] - W[0], W[2] - W[0])
                length = np.linalg.norm(n)
                if length < 1e-12:
                    return
                normals = np.tile(n / length, (3, 1))
            shade = (W, normals, mat)

        GL._fill_triangle(pts, zs, ws, color=mat["emissive"], vcolors=vcolors, uvs=uvs,
                          mipmaps=mipmaps, transparency=mat["transparency"], shade=shade)

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

        mat = GL._material(colors)

        # Matriz Model-View-Projection: leva pontos do espaço do objeto
        # (local) até o espaço de recorte (clip space). A divisão perspectiva
        # e o mapeamento para tela são feitos por vértice em _draw_triangle_3d.
        model = GL.transform_stack[-1]
        mvp = GL.perspective_matrix @ GL.view_matrix @ model

        # Agrupa de 9 em 9 valores (3 vértices x, y, z por triângulo). Cada
        # triângulo é facetado (uma normal por face).
        for t in range(0, len(point) - 8, 9):
            p0 = (point[t], point[t + 1], point[t + 2])
            p1 = (point[t + 3], point[t + 4], point[t + 5])
            p2 = (point[t + 6], point[t + 7], point[t + 8])
            GL._draw_triangle_3d(p0, p1, p2, mvp, mat, model)

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

        # Posição e orientação da câmera: usadas na iluminação (vetor até o
        # olho e direção da headlight).
        GL._eye = np.array(position, dtype=float)
        GL._cam_rot = camera_matrix[:3, :3]

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

        mat = GL._material(colors)
        P = np.array(point, dtype=float).reshape(-1, 3)

        tris = []
        offset = 0
        for count in stripCount:
            for j in range(count - 2):
                a, b, c = offset + j, offset + j + 1, offset + j + 2
                # Tiras de triângulo alternam a orientação dos vértices a
                # cada triângulo para manter a face consistente (sentido
                # horário/anti-horário) ao longo da tira.
                tris.append((a, b, c) if j % 2 == 0 else (b, a, c))
            offset += count
        GL._draw_indexed_mesh(P, tris, mat, smooth=GL.smooth_strips)

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

        mat = GL._material(colors)
        P = np.array(point, dtype=float).reshape(-1, 3)

        tris = []
        strip = []
        for idx in index:
            if idx == -1:
                # -1 marca o fim de uma tira; a próxima tira recomeça do zero
                strip = []
                continue
            strip.append(idx)
            if len(strip) >= 3:
                j = len(strip) - 3
                a, b, c = strip[j], strip[j + 1], strip[j + 2]
                # Alterna a orientação a cada triângulo para manter a face consistente
                tris.append((a, b, c) if j % 2 == 0 else (b, a, c))
        GL._draw_indexed_mesh(P, tris, mat, smooth=GL.smooth_strips)

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

        mat = GL._material(colors)
        model = GL.transform_stack[-1]
        mvp = GL.perspective_matrix @ GL.view_matrix @ model

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

            face_color = None  # cor única da face (0-255)
            if color and not colorPerVertex and mipmaps is None:
                ci = colorIndex[number] if colorIndex else number
                face_color = np.array(color[ci * 3:ci * 3 + 3]) * 255

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
                if face_color is not None:
                    vcs = [face_color] * 3
                elif per_vertex_color:
                    vcs = [t[1] for t in tri]
                else:
                    vcs = None
                GL._draw_triangle_3d(
                    tri[0][0], tri[1][0], tri[2][0], mvp, mat, model,
                    vcolors=vcs,
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

        mat = GL._material(colors)
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

        # Cada face tem a sua normal, então os cantos são repetidos por face
        # (6 faces x 4 cantos), cada face em 2 triângulos.
        faces = [
            ((0, 0, 1),  [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]),
            ((0, 0, -1), [(sx, -sy, -sz), (-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz)]),
            ((1, 0, 0),  [(sx, -sy, sz), (sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz)]),
            ((-1, 0, 0), [(-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz), (-sx, sy, -sz)]),
            ((0, 1, 0),  [(-sx, sy, sz), (sx, sy, sz), (sx, sy, -sz), (-sx, sy, -sz)]),
            ((0, -1, 0), [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)]),
        ]
        P, N, tris = [], [], []
        for normal, corners in faces:
            base = len(P)
            P.extend(corners)
            N.extend([normal] * 4)
            tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        GL._draw_indexed_mesh(P, tris, mat, vnormals=N)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        mat = GL._material(colors)
        stacks, slices = 20, 40  # divisões em latitude e longitude

        # Pontos da esfera unitária (também são as normais) e depois escala
        unit = []
        for i in range(stacks + 1):
            phi = math.pi * i / stacks  # 0 (polo norte) até pi (polo sul)
            for j in range(slices + 1):
                theta = 2 * math.pi * j / slices
                unit.append((math.sin(phi) * math.cos(theta), math.cos(phi),
                             math.sin(phi) * math.sin(theta)))
        unit = np.array(unit)

        tris = []
        for i in range(stacks):
            for j in range(slices):
                a = i * (slices + 1) + j
                b = a + slices + 1
                tris += [(a, b, a + 1), (a + 1, b, b + 1)]
        GL._draw_indexed_mesh(unit * radius, tris, mat, vnormals=unit)

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

        mat = GL._material(colors)
        n = 40  # divisões ao redor do eixo Y
        r, h = bottomRadius, height
        apex = (0.0, h / 2, 0.0)

        def ring(j):
            t = 2 * math.pi * j / n
            return (r * math.cos(t), -h / 2, r * math.sin(t))

        def side_normal(t):
            # normal da lateral inclinada: (h cos t, r, h sen t) normalizada
            v = np.array([h * math.cos(t), r, h * math.sin(t)])
            return v / np.linalg.norm(v)

        # Lateral: um triângulo por divisão; o vértice do topo usa a normal do meio
        P, N, tris = [], [], []
        for j in range(n):
            t0, t1 = 2 * math.pi * j / n, 2 * math.pi * (j + 1) / n
            base = len(P)
            P += [ring(j), ring(j + 1), apex]
            N += [side_normal(t0), side_normal(t1), side_normal((t0 + t1) / 2)]
            tris.append((base, base + 1, base + 2))
        GL._draw_indexed_mesh(P, tris, mat, vnormals=N)

        # Base (disco virado para baixo)
        P = [(0.0, -h / 2, 0.0)] + [ring(j) for j in range(n)]
        tris = [(0, 1 + j, 1 + (j + 1) % n) for j in range(n)]
        GL._draw_indexed_mesh(P, tris, mat, vnormals=[(0, -1, 0)] * len(P))

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

        mat = GL._material(colors)
        n = 40  # divisões ao redor do eixo Y
        hh = height / 2

        def circle(j, y):
            t = 2 * math.pi * j / n
            return (radius * math.cos(t), y, radius * math.sin(t))

        def out_normal(j):
            t = 2 * math.pi * j / n
            return (math.cos(t), 0.0, math.sin(t))

        # Lateral: dois triângulos por divisão, normais apontando para fora
        P, N, tris = [], [], []
        for j in range(n + 1):
            P += [circle(j, hh), circle(j, -hh)]
            N += [out_normal(j)] * 2
        for j in range(n):
            a = 2 * j
            tris += [(a, a + 1, a + 2), (a + 2, a + 1, a + 3)]
        GL._draw_indexed_mesh(P, tris, mat, vnormals=N)

        # Tampas (topo e base): discos em leque
        for y, ny in ((hh, 1), (-hh, -1)):
            P = [(0.0, y, 0.0)] + [circle(j, y) for j in range(n)]
            tris = [(0, 1 + j, 1 + (j + 1) % n) for j in range(n)]
            GL._draw_indexed_mesh(P, tris, mat, vnormals=[(0, ny, 0)] * len(P))

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # A headlight é uma luz direcional branca, intensidade 1, sem ambiente,
        # que aponta para onde a câmera olha; ela é montada em _active_lights().
        GL._headlight = bool(headlight)

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        d = np.array(direction, dtype=float)
        length = np.linalg.norm(d)
        if length < 1e-12:
            return
        GL._lights.append({"type": "dir", "dir": d / length,
                           "color": np.array(color, dtype=float),
                           "intensity": float(intensity),
                           "ambient": float(ambientIntensity)})

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        GL._lights.append({"type": "point", "loc": np.array(location, dtype=float),
                           "color": np.array(color, dtype=float),
                           "intensity": float(intensity),
                           "ambient": float(ambientIntensity)})

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

        # O relógio de cada sensor começa na primeira chamada, assim a animação
        # sempre parte do primeiro quadro-chave (fração 0).
        now = time.time()
        start = GL._timers.setdefault((cycleInterval, loop), now)
        elapsed = now - start

        if cycleInterval <= 0:
            return 0.0
        if loop:
            fraction_changed = (elapsed % cycleInterval) / cycleInterval
        else:
            fraction_changed = min(elapsed / cycleInterval, 1.0)  # roda um ciclo e para

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

        keys = np.array(key, dtype=float)
        vals = np.array(keyValue, dtype=float).reshape(-1, 3)
        n = len(keys)
        if n == 0 or len(vals) != n:
            return [0.0, 0.0, 0.0]

        f = float(set_fraction)
        if n == 1 or f <= keys[0]:
            return vals[0].tolist()
        if f >= keys[-1]:
            return vals[-1].tolist()

        # Se closed e primeira == última chave, a curva se fecha: as tangentes
        # das pontas consideram o vizinho do outro lado (Catmull-Rom).
        wrap = bool(closed) and n > 2 and np.allclose(vals[0], vals[-1])
        span = keys[-1] - keys[0]

        def tangent(j):
            """Derivada (valor por unidade de fração) na chave j."""
            if wrap and j == 0:
                t_prev, p_prev = keys[n - 2] - span, vals[n - 2]
                t_next, p_next = keys[1], vals[1]
            elif wrap and j == n - 1:
                t_prev, p_prev = keys[n - 2], vals[n - 2]
                t_next, p_next = keys[1] + span, vals[1]
            else:
                a, b = max(j - 1, 0), min(j + 1, n - 1)
                t_prev, p_prev, t_next, p_next = keys[a], vals[a], keys[b], vals[b]
            dt = t_next - t_prev
            return (p_next - p_prev) / dt if dt > 1e-12 else np.zeros(3)

        i = min(int(np.searchsorted(keys, f, side="right")) - 1, n - 2)
        dt = keys[i + 1] - keys[i]
        if dt <= 1e-12:
            return vals[i + 1].tolist()
        s = (f - keys[i]) / dt

        # Spline cúbica de Hermite entre as chaves i e i+1
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        value_changed = (h00 * vals[i] + h10 * dt * tangent(i)
                         + h01 * vals[i + 1] + h11 * dt * tangent(i + 1))

        return value_changed.tolist()

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

        keys = np.array(key, dtype=float)
        vals = np.array(keyValue, dtype=float).reshape(-1, 4)
        n = len(keys)
        if n == 0 or len(vals) != n:
            return [0, 0, 1, 0]

        f = float(set_fraction)
        if n == 1 or f <= keys[0]:
            return vals[0].tolist()
        if f >= keys[-1]:
            return vals[-1].tolist()

        def to_quat(r):
            """[x, y, z, ângulo] -> quatérnio unitário (x, y, z, w)."""
            axis = r[:3]
            length = np.linalg.norm(axis)
            if length < 1e-12:
                return np.array([0.0, 0.0, 0.0, 1.0])
            axis = axis / length
            half = r[3] / 2
            return np.array([*(axis * math.sin(half)), math.cos(half)])

        i = min(int(np.searchsorted(keys, f, side="right")) - 1, n - 2)
        dt = keys[i + 1] - keys[i]
        s = (f - keys[i]) / dt if dt > 1e-12 else 1.0

        q0, q1 = to_quat(vals[i]), to_quat(vals[i + 1])
        d = float(np.dot(q0, q1))
        if d < 0:  # q e -q são a mesma rotação; escolhe o caminho mais curto
            q1, d = -q1, -d
        if d > 0.9995:  # quase iguais: interpolação linear evita divisão por ~0
            q = q0 + s * (q1 - q0)
        else:  # SLERP: velocidade angular constante sobre a esfera unitária
            theta = math.acos(d)
            q = (math.sin((1 - s) * theta) * q0 + math.sin(s * theta) * q1) / math.sin(theta)
        q = q / np.linalg.norm(q)

        v = q[:3]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return [0, 0, 1, 0]
        angle = 2 * math.atan2(length, q[3])
        value_changed = [*(v / length), angle]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""