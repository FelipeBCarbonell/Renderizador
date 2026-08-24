import gpu

def draw_line(x0, y0, x1, y1, rgb):
    """Rasteriza um segmento de reta usando o algoritmo de Bresenham."""
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    from gl import GL  # import tardio evita import circular com gl.py

    while True:
        if 0 <= x0 < GL.width and 0 <= y0 < GL.height:
            gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, rgb)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy