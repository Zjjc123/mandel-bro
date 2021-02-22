import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from progress.bar import Bar
import matplotlib
from tree import Tree

from graphics import *

from colors import distinct_colors

diff_limit = 1

def reverse_colormap(cmap, name = 'my_cmap_r'):     
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = matplotlib.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

def diverges(c):
    return np.sqrt(c[0]**2 + c[1]**2) > 2

def complex_add(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1])

def complex_square(c):
    return (c[0]**2 - c[1]**2, 2*c[0]*c[1])

def iterate(a, b, max_it, input_z=0):
    count = 0
    z = (input_z, 0)
    c = (a, b)
    while (count < max_it):
        z = complex_add(complex_square(z), c)
        if diverges(z):
            return count
        count+=1
    return max_it
    
    
def simulate(xmin, xmax, ymin, ymax, max_it, max_depth, input_z=0):
    resolution = 50
    bar = Bar('Simulating', max=resolution)
    
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    n = np.zeros((resolution, resolution))
    xpos = np.zeros((resolution, resolution))
    ypos = np.zeros((resolution, resolution))

    # initialize low resolution tree
    for i in range(resolution):
        bar.next()
        for j in range(resolution):
            n[i,j] = iterate(x[j], y[i], max_it, input_z)
            xpos[i,j] = x[j]
            ypos[i,j] = y[i]
    
    new_children = np.stack((n.flatten(),xpos.flatten(),ypos.flatten()), axis=-1)
    #print(new_children.shape)
    # make a single layer tree with each grid/pixel
    mandelbrot_tree = Tree(children=new_children)
    
    #    *
    #   /|\
    #  1 2 3

    # refine tree
    for i in range(resolution-1):
        for j in range(resolution-1):
            # get the four corners of a grid cell
            corners = [n[i,j], n[i+1,j], n[i,j+1], n[i+1,j+1]]
            maxdiff = 0
            for k in range(4):
                for l in range(4):
                    if k!=l:
                        diff = np.abs(corners[k] - corners[l])
                        if diff > maxdiff:
                            maxdiff = diff
            #print(maxdiff)
            # if the difference is high enough that there are more details at that point
            index = np.ravel_multi_index((i,j), (resolution,resolution))
            if maxdiff > diff_limit:
                refine(mandelbrot_tree, index, x[j], y[i], x[j+1], y[i+1], n[i,j], max_it, max_depth, input_z, split=2)
                # make the tree refine
    bar.finish()
    
    visualize(mandelbrot_tree, xmin, xmax, ymin, ymax, resolution, split=2, max_it=max_it, max_depth = max_depth)

def refine(mandelbrot_tree, index, x1, y1, x2, y2, point, max_it, max_depth, input_z, split):
    #print(index)
    #print(len(mandelbrot_tree.children))
    #print(mandelbrot_tree.depth)
    if mandelbrot_tree.depth > max_depth:
        return

    n = np.zeros((split,split))
    xpos = np.zeros((split,split))
    ypos = np.zeros((split,split))

    x = np.linspace(x1, x2, split, endpoint=False)
    y = np.linspace(y1, y2, split, endpoint=False)

    for i in range(split):
        for j in range(split):
            n[i,j] = iterate(x[j], y[i], max_it, input_z)
            xpos[i,j] = x[j]
            ypos[i,j] = y[i]
    #print(x1, y1)
    #print(x2,y2)
    #print("\n")
    #print(xpos)
    #print(ypos)
    new_children = np.stack((n.flatten(),xpos.flatten(),ypos.flatten()), axis=-1)
    #print(new_children)
    #print(new_children.shape)
    mandelbrot_tree.children[index] = Tree(children=new_children, depth = mandelbrot_tree.depth + 1, val = point, x=x1, y=y1)
    #print(mandelbrot_tree.children[index].children)
    for i in range(split-1):
        for j in range(split-1):
            # get the four corners of an intersection
            corners = [n[i,j], n[i+1,j], n[i,j+1], n[i+1,j+1]]
            #print(corners)
            maxdiff = 0
            for k in range(4):
                for l in range(4):
                    if k!=l:
                        diff = np.abs(corners[k] - corners[l])
                        if diff > maxdiff:
                            maxdiff = diff
            # if the difference is high enough that there are more details at that point
            #if np.abs(y[i]) < 0.1:
                #print(maxdiff, x[j], y[i])
            new_index = np.ravel_multi_index((i,j), (split, split))
            if maxdiff > diff_limit:
                #print(x[j],y[i])
                #print(mandelbrot_tree.children[index].children)
                refine(mandelbrot_tree.children[index], new_index, x[j], y[i], x[j+1], y[i+1], n[i,j], max_it, max_depth, input_z, split=2)
"""        
def expand_array(n, expand_factor):
    new_n = np.zeros((n.shape[0] * expand_factor, n.shape[1] * expand_factor))
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            new_n[i*expand_factor:i*expand_factor+expand_factor,
             j*expand_factor:j*expand_factor+expand_factor] = n[i,j]

    return new_n

# bfs to convert a tree into a grid (duplicating for low res areas)
def get_array_from_tree(mandelbrot_tree, resolution, split):
    n = np.zeros((resolution, resolution))
    n_expand = 1
    queue = []
    queue.append(mandelbrot_tree)
    while queue:
        s = queue.pop(0)
        if s.depth < n_expand:
            expand_array(n, split)
        
        # use value
        for child in s.children:
            queue.append(child)
"""

def visualize(mandelbrot_tree, xmin, xmax, ymin, ymax, resolution, split, max_it, max_depth):
    cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}
    size = int(resolution*(split**max_depth))
    n = np.zeros((size, size))
    range_x = xmax-xmin
    range_y = ymax-ymin
    queue = []
    queue.append(mandelbrot_tree)
    colors = distinct_colors(max_it+1)
    while queue:
        s = queue.pop(0)
        if s.depth != 0:
            x1pos = int((s.x-xmin) * (size/range_x))
            y1pos = int((s.y-ymin) * (size/range_y))
            x2pos = int((s.x-xmin) * (size/range_x) + (size/resolution/(split**(s.depth-1))))
            y2pos = int((s.y-ymin) * (size/range_y) + (size/resolution/(split**(s.depth-1))))
            
            minx = np.max([0, x1pos])
            miny = np.max([0, y1pos]) 
            maxx = np.min([size-1, x2pos])
            maxy = np.min([size-1, y2pos])

            n[miny:maxy+1, minx:maxx+1] = s.val

        # use value
        for child in s.children:
            queue.append(child)


    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    my_cmap_r = reverse_colormap(my_cmap)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.matshow(n,interpolation=None, cmap=cm.get_cmap('Greys'),extent=[0,20,0,20])
    ax.matshow(n,interpolation=None, cmap=my_cmap_r,extent=[0,20,0,20])
    xlabels = np.round(np.linspace(xmin, xmax, 5),1)
    ylabels = np.round(np.linspace(ymax, ymin, 5),1)
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.set_aspect((ymax-ymin)/(xmax-xmin))
    plt.show()

"""
def visualize(mandelbrot_tree, xmin, xmax, ymin, ymax, resolution, split, max_it):
    win = GraphWin("Mandel Bro!", 1920, 1080)
    screensize = [1920, 1080]
    range_x = xmax-xmin
    range_y = ymax-ymin
    queue = []
    queue.append(mandelbrot_tree)
    colors = distinct_colors(max_it+1)
    while queue:
        s = queue.pop(0)
        if s.depth != 0:
            c = Rectangle(Point((s.x-xmin) * (screensize[0]/range_x), (s.y-ymin) * (screensize[1]/range_y)), 
            Point((s.x-xmin) * (screensize[0]/range_x) + (screensize[0]/resolution/(split**(s.depth-1))),
            (s.y-ymin) * (screensize[1]/range_y) + (screensize[1]/resolution/(split**(s.depth-1)))))
            c.setWidth(0)
            c.setOutline(colors[int(s.val)])
            c.setFill(colors[int(s.val)])
            c.draw(win)

        # use value
        for child in s.children:
            queue.append(child)

    win.getMouse() # Pause to view result
    win.close()    # Close window when done
    return
"""
simulate(-2, 2, -1, 1, 1000, 10, 0)
