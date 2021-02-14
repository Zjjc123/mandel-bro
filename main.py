# hi bestie don't be mad at me im sorry :( ;-; 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from progress.bar import Bar
import matplotlib

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

def iterate(a, b, maxit, input_z=0):
    count = 0
    z = (input_z, 0)
    c = (a, b)
    while (count < maxit):
        z = complex_add(complex_square(z), c)
        if diverges(z):
            return count
        count+=1
    return maxit
    
    
def simulate(xmin, xmax, ymin, ymax, maxit, resolution, input_z=0):
    bar = Bar('Simulating', max=resolution*resolution)
    
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    n = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            n[i,j] = iterate(x[j], y[i], maxit, input_z)
            bar.next()
    
    bar.finish()

    #n[n<maxit] = 0
    #n[n==maxit] = 1

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

simulate(-2, 2, -1, 1, 200, 1000, 0)
