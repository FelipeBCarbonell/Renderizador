def y_na_reta(p1, p2, x):
    dx = p2[0] - p1[0]
    if dx == 0:
        return None
    s = (p2[1] - p1[1]) / dx
    return p1[1] + s * (x - p1[0])

def no_intervalo_x(p1, p2, x):
    return min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])

def baixo(linha, ponto):
    p1, p2 = linha
    y_reta = y_na_reta(p1, p2, ponto[0])
    if y_reta is None:
        return False
    return (ponto[1] <= y_reta and no_intervalo_x(p1, p2, ponto[0]))

def alto(linha, ponto):
    p1, p2 = linha
    y_reta = y_na_reta(p1, p2, ponto[0])
    if y_reta is None:
        return False
    return ponto[1] >= y_reta and no_intervalo_x(p1, p2, ponto[0])

def inside(triangulo, ponto):
    p0, p1, p2 = triangulo
    arestas = [(p0, p1), (p1, p2), (p2, p0)]

    under = any(baixo(aresta, ponto) for aresta in arestas)
    over = any(alto(aresta, ponto) for aresta in arestas)

    return under and over
