**Liberty Metric Package**

```
from libMetric import liberty as lutil
import numpy as np
```


Liberty conversion
```
# load & convert CCS to JSON
lnode = lutil.read_lib('ccs.lib')
lutil.dump_json(lnode,out='ccs.json')
lnode.keys()

# load liberty from JSON
lnode = lutil.load_json('ccs.json')
lnode.keys()
```

Timing & power operation
```
# list cells in the liberary
[v for v in lnode['cell']]

# grab cell node by cell-name, e.g., 'ND2D1LVT'
cnode = lnode['cell']['ND2D1LVT']

# encapsulate all timing/power tables by timing-arc into a dataframe
lutT = lutil.get_cell_timing(cnode,todf=True)
lutP = lutil.get_cell_power(cnode,todf=True)
lutT.index # enumerate all lookup tables encapsulated by timing-arc
lutP.index # enumerate all lookup tables encapsulated by timing-arc
```

Lookup table, interpolation & regression
```
# lookup table interpolation, e.g., timing-arc ('A1,ZN,', 'combinational', 'cell_rise')
lut = lutT.loc[('A1,ZN,', 'combinational', 'cell_rise')]
y,x,v = map(np.array,lut.values) # unpack values as numpy array

# timing interpolation based on the specified transition & load
lutil.table_lookup(lut,trans=0.0207,load=0.0010072,dflag=True)

# LS regression & prediction
lutil.lut2lsCoeff(lut.to_dict(),trans=0.03,load=0.0017,dflag=True)
```

Data visualization API
```
cnode = lnode['cell']['DFCNQD1LVT']

lutT = lutil.get_cell_timing(cnode,todf=False) # grab all timing tables in JSON
lutil.plot_lut(lutT,keys=[('CP,Q,', 'rising_edge', 'cell_rise'),
            ('CP,Q,', 'rising_edge', 'cell_fall')],xylabel=('load','trans'))

lutil.plot_lut(lutT,keys=[('CP,D,CDN', 'setup_rising', 'rise_constraint'),
            ('CP,D,CDN', 'hold_rising', 'rise_constraint')],xylabel=('clock','data'))
```






---



**GRO Compiler Package**

```
from GRO import ROCompiler 
import sys

if __name__ == '__main__':
    argv = sys.argv
else: # test mode
    argv = ['C:/Home/Projects/Pypi/DTCO/GRO/gro.py',
            '-config','C:/Home/Projects/Pypi/DTCO/GRO/demo/config_demo.f',
            '-outDir','C:/Home/Projects/Pypi/DTCO/GRO/demo/RO_demo',
            '-target','TT']

# init GRO instance
gro = ROCompiler()

# parse command line
code,pdata = gro.parseArguments(argv)

# update pdata from config
if pdata.get('configFile')!=None:
    cfg = gro.loadConfig(pdata['configFile'])
if pdata.get('initProj')==True: # create RO project directory
    gro.initProjectDirectory()
    gro.initMakefile()
if pdata.get('initLib')==True:
    gro.initLibJSON() # build library JSON DB
if pdata.get('buildRO')==True: # generate RO design and the successive DC, synthesis, vsim, SPICE environments
    gro.commitConfig() # start from liberty JSON without initLibJSON
    gro.compileGRO()
if pdata.get('lpe')!=None:
    gro.genSPICESim()
```

---

**Design & Technology C-optimization**

Process Uniformity & OCV Analysis

Machine-learning Framework

