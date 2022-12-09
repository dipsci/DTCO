# Liberty Metric Extraction Training Matrial
# Topic:
#     1. query and underatand timing/power table through API
#     2. visualize timing/power table through API

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

import sys
sys.path.append('C:/Home/Projects/Pypi/DTCO')
from libMetric import liberty as lutil

#import pandas as pd
#pd.set_option('expand_frame_repr', False)

##########################################################################################
#%% build libDB
import glob
jsonL = glob.glob(f'C:/Home/Projects/Pypi/DTCO/libMetric/demo/JSON/*.json')
libDB = {}
for json in jsonL:
    lnode = lutil.load_json(json)
    lname = lnode['library']
    libDB[lname] = lnode

[v for v in libDB]

#%% grab/encapsulate all timing tables into a dictionary with key values as (arc,ttype,ctype)
lnode = libDB['libertydemo']
[v for v in lnode['cell']]

cnode = lutil.get_cell(lnode,'ND2D1LVT')
lutT = lutil.get_cell_timing(cnode) # all timing lut
[v for v in lutT]

lutP = lutil.get_cell_power(cnode) # all power lut
[v for v in lutP]

#%% timing table visualization
def plot_lut(lutT,keys=None,title=None,xylabel=('load','tran'),ax=None):
    '''lutT := a dictionary encapsulates lut with key values as ('arc','ttpy','ctyp') form
       keys := a key or a list of key values described above 
       wherein lut is a dictionary as {'index_1':[...],'index_2':[...],'values':[...]}'''
    if isinstance(keys,list)==False:
        keys = [keys]
    f = plt.figure()
    f.set_size_inches(8,7)
    ax = Axes3D(f)
    for k in keys:
        arc,ttype,ctype = k
        lut = lutT[k]
        if len(lut)==2:
            y,z = map(np.array,lut.values())
            ax.plot(y,z,marker='o',ms=5,zdir='x',zs=0,label=k)
            ax.view_init(0,0)
        else:
            y,x,z = map(np.array,lut.values())
            z = z.reshape(len(y),len(x))
            if 'hold' in str(k):
                z = -z # inverse hold time
            gx,gy = np.meshgrid(x,y)
            px = np.arange(0,x[-1]+x[-1]/50,(x[-1]/50)) # col
            py = np.arange(0,y[-1]+y[-1]/50,(y[-1]/50)) # row
            gpx,gpy = np.meshgrid(px,py) # fine-grained grid
            gpz = interpolate.interp2d(x,y,z,kind='linear')(px,py)
            p = ax.scatter(gx.ravel(),gy.ravel(),z.ravel(),s=10,label=k)
            ax.plot_surface(gpx,gpy,gpz,alpha=0.4,color=p.get_facecolors()[0]) #cmap=plt.cm.viridis)
            ax.view_init(30,250)
    ax.set_xlabel(f'index_2:{xylabel[0]}')
    ax.set_ylabel(f'index_1:{xylabel[1]}')
    ax.set_zlabel('values')
    plt.legend()
    plt.show()

cnode = lutil.get_cell(lnode,'DFCNQD1LVT')
lutT = lutil.get_cell_timing(cnode)

# visualization of lut, check the distribution of index range
plot_lut(lutT,keys= ('CP,CP,CDN&D', 'min_pulse_width', 'rise_constraint'))
plot_lut(lutT,keys=('CP,Q,','rising_edge','cell_rise'))

plot_lut(lutT,keys=[('CP,D,CDN','setup_rising','rise_constraint'),
                    ('CP,D,CDN','hold_rising', 'rise_constraint')],xylabel=('clock','data'))

plot_lut(lutT,keys=[('CP,D,CDN','setup_rising','fall_constraint'),
                    ('CP,D,CDN','hold_rising', 'fall_constraint')],xylabel=('clock','data'))

#%% timing table visualization with API
cnode = lutil.get_cell(lnode,'DFCNQD1LVT')
lutT = lutil.get_cell_timing(cnode)
[v for v in lutT]

# API
lutil.plot_lut(lutT,keys=('CP,Q,', 'rising_edge', 'cell_rise'))
lutil.plot_lut(lutT,keys=('CP,D,CDN', 'setup_rising', 'rise_constraint'),xylabel=('clock','data'))

# constraint rise & fall comparison
keyM = [('CP,CP,CDN&D', 'min_pulse_width', 'rise_constraint'),
        ('CP,CP,CDN&D', 'min_pulse_width', 'fall_constraint')]
lutil.plot_lut(lutT,keys=keyM)

# delay rise & fall comparison
keyL = [('CP,Q,', 'rising_edge', 'cell_rise'),
        ('CP,Q,', 'rising_edge', 'cell_fall')]
lutil.plot_lut(lutT,keys=keyL)

# setup & hold
keyC = [('CP,D,CDN', 'hold_rising', 'rise_constraint'),
        ('CP,D,CDN', 'setup_rising', 'rise_constraint')]
lutil.plot_lut(lutT,keys=keyC,xylabel=('clock','data'))

keyC = [('CP,D,CDN', 'hold_rising', 'fall_constraint'),
        ('CP,D,CDN', 'setup_rising', 'fall_constraint')]
lutil.plot_lut(lutT,keys=keyC,xylabel=('clock','data'))

#%%
def plot_cell_timing(cnode,arc,ctype,xylabel=('load','trans')):
    lutT = lutil.get_cell_timing(cnode,ctype=ctype)
    keyL = [(a,t,c) for a,t,c in lutT if ctype in c and arc==a]
    lutil.plot_lut(lutT,keys=keyL)
    
plot_cell_timing(cnode,arc='CP,Q,',ctype='cell')
plot_cell_timing(cnode,arc='CP,D,CDN',ctype='rise_constraint',xylabel=('clock','data'))

#%% timing table visualization API
lutil.plot_cell_timing(cnode,arc='CP,Q,',ctype='cell')
lutil.plot_cell_timing(cnode,arc='CP,D,CDN',ctype='rise_constraint',xylabel=('clock','data'))

