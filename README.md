**Preliminary**

Liberty Metric Extraction

Grid RO Compiler

---

**Copernic System**

Generated Model for Virtual Silicon

WAT Analysis

CP Analysis


---

**Design & Technology Co-optimization**

Process Uniformity & OCV Analysis

Machine-learning Framework

Binning Strategy

---


**Installation**
```
pip install DTCO

```

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
    argv = ['.',
            '-config','config_demo.f',
            '-outPath','RO_demo',
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
**DTCO Platform**

```
from copernic import dtco

# load the pre-trained model and generate 300 virtual silicon wafers
dt = dtco.genFakeData('/content/gmodel_C10.pkl',num=300).set_index(['WID'])

# CP data
dtco.batchFeature(dt,feature='SIDD',num=10,ncol=5,dtype='2d')
dtco.batchFeature(dt,feature='ROu',num=10,ncol=5,dtype='2d')

# WAT data
dtco.batchFeature(dt,feature='VTS_ULVT_N',num=10,ncol=5,dtype='3d')
dtco.batchFeature(dt,feature='VTS_ULVT_P',num=10,ncol=5,dtype='3d')

# # batch visualization on web UI
dtco.batchFeaturePlotly(dt,feature='SIDD',widL=range(1,9),ncol=4)

# all wafer scatter, 200 sub samples steps
dtco.featureScatter(dt.iloc[::200],wid=None,fx='ROu',fy='SIDD',s=2,alpha=0.1)

# single wafer scatter and die XY location on the feature surface
dtco.featureScatter(dt,wid=1,fx='ROu',fy='SIDD',s=10,alpha=0.5)

# feature surface
dtco.featureSurface(dt,wid=1,feature='SIDD',sigma=2.5)

# visualization on web UI
dtco.featureSurfacePlotly(dt,wid=1,feature='SIDD',sigma=2.5)

```
Productivity Enhancement
```
dw = dtco.waferSort(dt,itemL=['ROu','SIDD'],nsize=300)

# PCM density 3D
binx,biny,H = dtco.pcmDensity3D(dt,fx='ROu',fy='SIDD',sigma=2.5)

# PCM density 2D
binx,biny,H = dtco.pcmDensity2D(dt,fx='ROu',fy='SIDD',sigma=2.5)

# yield assessment
dtco.pcmDensity(dt,fx='ROu',fy='SIDD',sigma=2.5,percentiles=[1,5,10,20,30])

# 2D binning evaluation, boundary and bin yield
hbin = dtco.pcmBinning(dt,fx='ROu',fy='SIDD',sigma=2.5,bins=(4,6))

# compromise design recipe/strategy and yield assessment
dtco.pcmYieldAssessment(dt,fx='ROu',fy='SIDD',percentiles=[10,20,30],sigma=2.5)

```

---

