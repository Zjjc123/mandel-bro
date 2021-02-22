import colorsys

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def distinct_colors(n):
    n = n-1
    HSV_tuples = [(x/n, 1, 1) for x in range(n)]
    rgb = []
    for c in HSV_tuples:
        rgb.append(rgb2hex(int(colorsys.hsv_to_rgb(*c)[0]*255), int(colorsys.hsv_to_rgb(*c)[1]*255), int(colorsys.hsv_to_rgb(*c)[2]*255)))
    rgb.append("#000000")
    return rgb
