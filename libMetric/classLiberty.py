"""
Liberty Metric Extraction v0.8 

Copyright (C) 2022, by DipSci Technology Corpopation, LTD.
This software tool is independently developed and owned by DipSci Technology Co., Ltd.
It is freely licensed to industry and academia to use, share, reproduce, or improve, provided 
this copyright notice is retained or the initial creater is listed on the contributors list.
Derivatives developed for profit based on this software tool must still include this copyright 
notice or credit the initial creater on the contributors list.

Revision history:
Hockchen 2020/03/12 init
Hockchen 2022/02/25 libmetric class
Hockchen 2022/05/26 add operating_conditions
Hockchen 2022/07/20 ecsm model
"""
import pandas as pd
import numpy as np
import scipy.linalg
import gzip, re, json 
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

class Liberty:
    def __init__(self):
        print('init class Liberty ...')
        self.content = []
        self.libDec = {
            'library':('value',''),   
            'technology':('value',''),
            'delay_model':('value',''),
            'slew_derate_from_library':('value',''),
            'nom_process':('value',''),
            'nom_temperature':('value',''),
            'nom_voltage':('value',''),
            'voltage_map':('value',''),
            'capacitive_load_unit':('value',''),
            'voltage_unit':('value',''),
            'current_unit':('value',''),
            'time_unit':('value',''),
            'operating_conditions':('callback',self.parse_as_operating_conditions),
            #'operating_conditions':('value',''),
            'wire_load':('callback',self.parse_as_undefgroup),
            'wire_load_selection':('callback',self.parse_as_undefgroup),
            'lu_table_template':('callback',self.parse_as_undefgroup),
            'power_lut_template':('callback',self.parse_as_undefgroup),
            'normalized_driver_waveform':('callback',self.parse_as_undefgroup),
            'cell':('callback',self.parse_as_cell),
            '_L':('bos',''),
            '_R':('eos','')
        }    
        self.opcDec = { # 2022/05/26 for operating_conditions
            'process':('value',''),
            'temperature':('value',''),
            'voltage':('value',''),
            'tree_type':('value',''),
            '_L':('bos',''),
            '_R':('eos','')
        }
        self.cellDec = {
            'area':('value',0),
            'antenna_diode_type':('waive',0),
            'cell_footprint':('value',0),
            'cell_leakage_power':('value',0),
            'is_level_shifter':('value',0),
            'is_clock_cell':('value',0),
            'is_clock_isolation_cell':('value',0),
            'is_isolation_cell':('value',0),
            'always_on':('value',0),
            'switch_cell_type':('value',''),
            'retention_cell':('value',''),
            'cell_description':('waive',''),
            'user_function_class':('waive',''),
            'level_shifter_type':('value',''),
            'input_voltage_range':('value',''),
            'output_voltage_range':('value',''),
            'dont_use':('value',''),
            'dont_touch':('value',''),
            'define':('waive',''),
            'clock_gating_integrated_cell':('value',''),
            'scaling_factors':('callback',self.parse_as_undefgroup),
            'intrinsic_parasitic':('callback',self.parse_as_undefgroup),
            'dynamic_current':('callback',self.parse_as_undefgroup),
            'test_cell':('callback',self.parse_as_undefgroup),
            'statetable':('callback',self.parse_as_undefgroup),
            'pg_pin':('callback',self.parse_as_undefgroup),
            'leakage_power':('callback',self.parse_as_leakage_power),
            'leakage_current':('callback',self.parse_as_undefgroup),
            'ff':('callback',self.parse_as_undefgroup),
            'ff_bank':('callback',self.parse_as_undefgroup),
            'bundle':('callback',self.parse_as_bundle),
            'latch':('callback',self.parse_as_undefgroup),
            'latch_bank':('callback',self.parse_as_undefgroup),
            'clock_condition':('callback',self.parse_as_undefgroup),
            'clear_condition':('callback',self.parse_as_undefgroup),
            'preset_condition':('callback',self.parse_as_undefgroup),
            'retention_condition':('callback',self.parse_as_undefgroup),
            'dc_current':('callback',self.parse_as_undefgroup),
            'pin':('callback',self.parse_as_pin),
            '_L':('bos',''),
            '_R':('eos','')
        }
        self.bundDec = {
            'members':('waive',''),
            'direction':('waive',''),
            'functione':('waive',''),
            'power_down_function':('waive',''),
            'pin':('callback',self.parse_as_pin), 
            '_L':('bos',''),
            '_R':('eos','')
        }
        self.pinDec = {
            'direction':('value',0),
            'capacitance':('value',0),
            'driver_type':('value',''),
            'three_state':('value',''),
            'clock':('value',''),
            'clock_gate_clock_pin':('value',''),
            'clock_gate_enable_pin':('value',''),
            'clock_gate_test_pin':('value',''),
            'clock_gate_out_pin':('value',''),
            'state_function':('waive',''),
            'internal_node':('waive',''),
            'nextstate_type':('waive',''),
            'driver_waveform_fall':('waive',''),
            'driver_waveform_rise':('waive',''),
            'related_ground_pin':('waive',''),
            'related_power_pin':('waive',''),
            'related_bias_pin':('waive',''),
            'rise_capacitance':('value',''),
            'fall_capacitance':('value',''),
            'power_down_function':('waive',''),
            'function':('value',''),
            'max_capacitance':('waive',''),
            'min_capacitance':('waive',''),
            'max_transition':('value',0),
            'antenna_diode_related_ground_pins':('waive',''),
            'internal_power':('callback',self.parse_as_internal_power),
            'timing':('callback',self.parse_as_timing),
            'receiver_capacitance':('waive',self.parse_as_undefgroup), # CCSN
            '_L':('bos',''),
            '_R':('eos','')
        }    
        self.timeDec={
            'related_pin':('value',''),
            'timing_sense':('value',''),
            'timing_type':('value',''),
            'sdf_cond':('value',''),
            'when':('value',''),
            'cell_rise':('callback',self.parse_as_lut),
            'cell_fall':('callback',self.parse_as_lut),
            'rise_transition':('callback',self.parse_as_lut),
            'fall_transition':('callback',self.parse_as_lut),
            'rise_constraint':('callback',self.parse_as_lut),
            'fall_constraint':('callback',self.parse_as_lut),
            'output_current_fall':('waive',self.parse_as_undefgroup), # CCSN
            'output_current_rise':('waive',self.parse_as_undefgroup), # CCSN
            'receiver_capacitance1_rise':('waive',self.parse_as_undefgroup), # CCSN
            'receiver_capacitance2_rise':('waive',self.parse_as_undefgroup), # CCSN
            'receiver_capacitance1_fall':('waive',self.parse_as_undefgroup), # CCSN
            'receiver_capacitance2_fall':('waive',self.parse_as_undefgroup), # CCSN
            '_L':('bos',''),
            '_R':('eos','')
        }
        self.powerDec = {
            'rise_power':('callback',self.parse_as_lut),
            'fall_power':('callback',self.parse_as_lut),
            'related_pin':('value',''),
            'related_pg_pin':('value',''),
            'when':('value',''),
            '_L':('bos',''),
            '_R':('eos','')
        }
        self.leakageDec = {
            'value':0,
            'related_pg_pin':'',
            'when':''
        }

    # Remove C-style /*comments*/ from a string
    def remove_comments(text):
        p = r'/\*[^*]*\*+([^/*][^*]*\*+)*/|("(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'|.[^/"\'\\]*)'
        return ''.join(m.group(2) for m in re.finditer(p,text,re.M|re.S) if m.group(2))

    def parse_as_undefgroup(self,ti,args=''):
        if self.content[ti+1]!='_L': # simple or complex attributes
            return ti
        size = len(self.content)
        ii = ti
        state = 0
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            #print(f'skip>'+token)
            if token=='_L':
                state += 1
            elif token=='_R':
                state -= 1
                if state==0:
                    break
            ii += 1
        return ii

    def parse_as_bundle(self,ti,cnode):
        #iname = re.split(r'[()]',self.content[ti])[1]
        #print(f'bundle> {iname}')
        size = len(self.content)
        ii = ti+1
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            #print(f'  bundle token> {token}')
            if token=='':
                ii+=1
                continue
            if self.bundDec.get(token)!=None:
                t,func = self.bundDec[token]
                if t=='callback':
                    ii = func(ii,cnode)
                elif t=='value':
                    pass
                    #val = re.split(r'[:]',tokens)[1]
                    #print(f'  bundle> {token}={val}')
                elif t=='waive':
                    #print(f'  bundle waive> {token}')
                    ii = self.parse_as_undefgroup(ii,token)
                elif t=='bos': # begin of scope
                    ii += 1
                    continue
                else: # end of scope
                    break
            else:
                ii = self.parse_as_undefgroup(ii,token)
            ii+=1
        return ii
    
    def parse_as_operating_conditions(self,ti,libnode):
        name = re.split(r'[()]',self.content[ti])[1].strip('"')
        node = {}
        size = len(self.content)
        ii = ti+1
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            #print(f'opc token> {token}')
            if token=='':
                ii += 1
                continue
            if self.opcDec.get(token)!=None:
                t,func = self.opcDec[token]
                if t=='callback':
                    ii = func(ii,node)
                elif t=='value':
                    val = re.split(r'[():]',tokens)[1]
                    node[token]=val.strip('"')
                    #print(f'  opc> {token} = {val}')
                elif t=='bos' or t=='waive': # begin of scope
                    ii += 1
                    continue
                else: # end of scope
                    break 
            else:
                ii = self.parse_as_undefgroup(ii,token)
            ii += 1
        libnode['operating_conditions'][name] = node
        return ii

    def parse_as_pin(self,ti,cnode):
        iname = re.split(r'[()]',self.content[ti])[1]
        #print(f'  pin> {iname}')
        size = len(self.content)
        ii = ti+1
        tnode = {} # returned timing/power table
        inode = {'name':iname} # temporary parse data
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            if token=='':
                ii += 1
                continue
            if self.pinDec.get(token)!=None:
                t,func = self.pinDec[token]
                if t=='callback':
                    ii,tnode = func(ii)
                    if inode.get(token)==None:
                        inode[token] = []
                    inode[token].append(tnode)
                elif t=='value':
                    val = re.split(r'[:]',tokens)[1]
                    inode[token] = val.strip('"')
                    #print(f'  pin> {token} = {val}')
                elif t=='waive':
                    ii = self.parse_as_undefgroup(ii,token)
                elif t=='bos': # begin of scope
                    ii += 1
                    continue
                else: # end of scope
                    break
            else:
                ii = self.parse_as_undefgroup(ii,token)
            ii += 1
        cnode['pin'][iname] = inode
        return ii

    def parse_as_lut(self,ti,vType):
        size = len(self.content)
        ii = ti+1
        vnode = {} # temporary parse data
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            if token=='' or token=='_L':
                ii += 1
                continue
            elif token=='_R': # enf of scope
                break
            if token[:6]=='index_': #token=='index_1' or token=='index_2' or token=='index_3':
                vt = re.split(r'[():"]',tokens)[2]
                vl = vt.split(',')
                vnode[token]=list(map(float,vl))
                #print(f'    {token}={vl}')
            elif token=='values':
                vt = "".join(re.split(r'[():"]',tokens)[1:])
                vl = vt.split(',')
                #vnode['values']=vl #np.array(vl,dtype='float64')
                #!NOTE! JSON doesn't support numpy.array serializaation 
                vnode['values']=list(map(float,vl))
            else:
                #print(f'    LUT unknow> {token}')
                ii = self.parse_as_undefgroup(ii,token) # ecsm_waveform_set 2022
            ii += 1
        return ii,vnode

    def parse_as_timing(self,ti):
        size = len(self.content)
        ii=ti+1
        tnode = {} # temporary parse data
        vnode = {}
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            if token=='': 
                ii += 1
                continue
            if self.timeDec.get(token)!=None:
                t,func = self.timeDec[token]
                if t=='callback':
                    ii,vnode = func(ii,token)
                    tnode[token] = vnode
                elif t=='value':
                    val = re.split(r'[():]',tokens)[1]
                    tnode[token] = val.strip('"')
                    #print(f'    timing> {token} = {val}')
                elif t=='waive':
                    ii = self.parse_as_undefgroup(ii,token)
                elif t=='bos': # begin of scope
                    ii += 1
                    continue
                else: # end of scope
                    break
            else:
                ii = self.parse_as_undefgroup(ii,token)
            ii += 1
        return ii,tnode

    def parse_as_internal_power(self,ti):
        size = len(self.content)
        ii = ti+1
        pnode={} # temporary parse data
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            if token=='':
                ii += 1
                continue
            if self.powerDec.get(token)!=None:
                t,func = self.powerDec[token]
                if t=='callback':
                    ii,vnode = func(ii,token)
                    pnode[token] = vnode
                elif t=='value':
                    val = re.split(r'[():]',tokens)[1]
                    pnode[token] = val.strip('"')
                    #print(f'    int_power> {token} = {val}')
                elif t=='bos': # begin of scope
                    ii+=1
                    continue
                else: # end of scope
                    break
            else:
                ii = self.parse_as_undefgroup(ii,token)
            ii += 1
        return ii,pnode

    def parse_as_leakage_power(self,ti,cnode):
        node = {}
        ei = self.content.index('_R',ti)
        ii = ti
        while ii<ei:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            if self.leakageDec.get(token)!=None:
                val = re.split(r'[():]',tokens)[1]
                node[token] = val.strip('"')
            ii += 1
        cnode['leakage_power'].append(node)
        return ei

    def parse_as_cell(self,ti,libnode):
        cname = re.split(r'[()]',self.content[ti])[1].strip('"')
        #print(f'cell> {cname}')
        cnode={
            'name':cname,
            'pin':{},
            'leakage_power':[]
        }
        size = len(self.content)
        ii = ti+1
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0]
            #print(f"  cell>{token}")
            if token=='':
                ii += 1
                continue
            if self.cellDec.get(token)!=None:
                t,func = self.cellDec[token]
                if t=='callback':
                    ii = func(ii,cnode)
                elif t=='value':
                    val = re.split(r'[():]',tokens)[1]
                    cnode[token]=val.strip('"')
                    #print(f'cell> {token} = {val}')
                elif t=='bos' or t=='waive': # begin of scope
                    ii += 1
                    continue
                else: # end of scope
                    break 
            else:
                ii = self.parse_as_undefgroup(ii,token)
            ii += 1
        libnode['cell'][cname] = cnode
        return ii

    # load liberty file
    def read_lib(self,lib,gzFlag=False):
        print('load liberty %s ...' % lib)
        if gzFlag==True:
            with gzip.open(lib,'rt') as f: 
                text = f.read()
        else:
            with open(lib,'r') as f:
                text = f.read()
    
        text = re.sub(r'//.*\n|\\\n|[ \t]+','',text) # remove white space
        text = re.sub(r'/\*.*\*/','',text) # remove comments
        text = re.sub(r'{','@_L@',text) # replace left bracket
        text = re.sub(r'}','@_R@',text) # replace right bracket
        self.content = re.split(r'[@;\n]+',text)
        ii,size = 0,len(self.content)
        libnode = {'cell':{},'operating_conditions':{}}
        while ii<size:
            tokens = self.content[ii]
            token = re.split(r'[():]',tokens)[0] 
            #print(f'lib token> {token}')
            if token=='': 
                ii += 1
                continue
            if self.libDec.get(token)!=None:
                t,func = self.libDec[token]
                if t=='callback':
                    ii = func(ii,libnode)
                elif t=='value':
                    val = re.split(r'[():]',tokens)[1]
                    libnode[token] = val.strip('"')
                elif t=='bos' or t=='waive':  # begin of scope
                    ii += 1
                    continue
                else: # end of scope
                    break
            else: # waive unknow token as attribute statement
                ii = self.parse_as_undefgroup(ii,token)
            ii += 1
        return libnode
    
    # convert JSON string into dictionary
    def load_json(self,fname):
        print('load library %s ...' % (fname))
        with open(fname,mode='r') as f:
            jnode = f.read()
        libnode = json.loads(jnode)
        return libnode
    
    def load_metric(self,fname): # to be included in classMetric
        if 'pkl' in fname.split('.')[-1]: # from Pickle
            df = pd.read_pickle(fname).set_index(['cell'])
        else: # from LSC CSV
            dc = pd.read_csv(fname).set_index(['cell'])
            nan = None #np.nan
            dfL = []
            for cell in dc.index:
                d = dc.loc[cell].copy()
                m = eval(d['metrics']) # convert metric to dict
                #m = pd.DataFrame(m).T # convert metric to Dataframe
                d['metrics'] = m.copy()
                dfL += [d]
            df = pd.DataFrame(dfL)
            df.index.name = 'cell'
        return df
    
    # basic query functions
    def get_cell(self,libnode,cname):
        cnode = {}
        cells = libnode['cell']
        for ckey in cells.keys():
            if ckey==cname:
                cnode = cells[cname]
                break
        return cnode
    
    def get_pins(self,cnode,pname='.*',direction='.*'):
        '''return pin data encapsulated in a dictionary form, e.g. {'ZN':data}'''
        pins = cnode['pin']
        pnodes = {}
        for pkey in pins.keys():
            pnode = pins[pkey]
            if bool(re.match(pname,pkey)) and bool(re.match(direction,pnode['direction'])):
                pnodes[pkey] = pnode
        return pnodes

    # guery data with cell level
    def get_cells(self,libnode,cname_re='-',cfp_re='-'):
        cnodes = []
        cells = libnode['cell']
        for ckey in cells.keys():
            cnode = cells[ckey]
            fp = cnode['cell_footprint'].strip('"') if cnode.get('cell_footprint')!=None else ''
            if bool(re.match(cfp_re,fp)) or bool(re.match(cname_re,ckey)):
                cnodes.append(cnode)
        return cnodes # list of cell nodes

    # ttype := timing type (optional)
    # ctype := constraint/table type (must)
    # ttype:  ctype
    # rising_edge:  {cell_rise, cell_fall, rise_transition, fall_transition}
    # falling_edge: {cell_rise, cell_fall, rise_transition, fall_transition}
    # setup_rising: {rise_constraint, fall_constraint}
    # hold_rising:  {rise_constraint, fall_constraint}
    def get_cell_timing(self,cnode,ttype=None,ctype=None,todf=False,no_tri_state=True):
        '''ctype := constraint type, optional, {cell_rise|cell_fall|rise_constraint|fall_constraint}
           return lut encapsulated in a dictionary form, e.g. {('A1,Z,','cell_rise'):lut}
           where lut is a dictionary as {'index_1':[...],'index_2':[...],'values':[...]}'''
        dnodes = {}
        incL = set(['rise_constraint','fall_constraint','cell_rise','cell_fall','rise_transition','fall_transition'])
        #excL = set(['timing_sense','timing_type','related_pin','sdf_cond','when'])
        for pin in cnode['pin'].keys():
            inode = cnode['pin'][pin]
            if inode.get('timing')==None:
                continue
            tables = inode['timing']
            for tnode in tables:
                if ttype!=None:
                    if ttype not in tnode['timing_type']:
                        continue
                cond = tnode['when'] if tnode.get('when')!=None else ''
                arc = ",".join([tnode['related_pin'],pin,cond])
                kL = set(tnode.keys())&incL
                for ckey in kL:
                    if ctype!=None and ctype not in ckey:
                        continue
                    tkey = tnode.get('timing_type')
                    if no_tri_state and 'disable' in str(tkey): # waive 3-state disable
                        continue
                    lut = tnode[ckey]
                    dnodes[arc,tkey,ckey] = lut
        if todf==True: # convert to DataFrame
            dnodes = pd.DataFrame.from_dict(dnodes,orient='index')
            dnodes.index.names = ['arc','ttype','ctype']
        return dnodes # dict {('iPin,oPin,when',ttype,ctype): lut}

    def get_cell_power(self,cnode,related_pg_pin='VDD',ctype=None,todf=False):
        '''ctype := constraint type, {fall_power|rise_power}
           return tables encapsulated in a dictionary form, e.g. {('A1,Z,','rise_power'):lut}'''
        dnodes = {}
        for pin in cnode['pin'].keys():
            inode = cnode['pin'][pin]
            if inode.get('internal_power')==None:
                continue
            tables = inode['internal_power']
            for tnode in tables:
                pg = tnode['related_pg_pin'].strip('"') if tnode.get('related_pg_pin')!=None else ''
                if bool(re.match(pg,related_pg_pin))==False:
                    continue
                rinp = tnode['related_pin'].strip('"') if tnode.get('related_pin')!=None else ''            
                cond = tnode['when'] if tnode.get('when')!=None else ''
                arc = ",".join([rinp,pin,cond])
                # empty 'rinp' indicates an input pin power
                for k in ['fall_power','rise_power']:
                    if ctype!=None and ctype not in k:
                        continue
                    dnodes[arc,k] = tnode[k] # lut
        if todf==True: # convert to DataFrame
            dnodes = pd.DataFrame.from_dict(dnodes,orient='index')
            dnodes.index.names = ['arc','ctype']
        return dnodes # dict {'iPin:oPin,when' : lut}

    # basic timing/power lookup functions
    def table_lookup(self,lut,trans,load,dflag=False):
        if lut.get('index_1')==None:
            return 0
        d = self.lut2df(lut)
        if len(lut)==2:
            ip = interpolate.interp1d(d.index.values,d.values.ravel(),kind='linear',fill_value='extrapolate')
            v = ip(trans)
        else:
            ip = interpolate.interp2d(d.columns.values,d.index.values,d.values,kind='linear')
            v = ip(load,trans)[0]
        if dflag:
            print(d)
            print(f'interpolation({trans},{load})= {v}')
        return v # interpolation data

    def lookup_cell_pincap(self,cnode,dflag=False):
        cL = []
        for k in cnode['pin'].keys():
            inode = cnode['pin'][k]
            if inode.get('capacitance') is None:
                continue
            cap = float(inode['capacitance'])
            cL += [cap]
            if dflag:
                print(f'pin({k})={cap}')
        cap = np.array(0.0) if len(cL)==0 else np.array(cL)
        chk = cap.std()*100/cap.mean()
        if chk>80:
            print(f'WARN! capacitance variation among {len(cL)} pins {chk:.2f}% > 80% @{cnode["name"]}')
        return cap.mean()

    def lookup_cell_leakage(self,cnode,related_pg_pin='VDD',dflag=False):
        vL = []
        for pnode in cnode['leakage_power']:
            cond = pnode['when'] if pnode.get('when')!=None else ''
            if pnode['related_pg_pin'].strip('"')==related_pg_pin:
                v = float(pnode['value'])
                vL += [v]
                if dflag:
                    print(f'leakage@{cond}= {v}')
        return np.array(vL).mean()

    def lookup_cell_timing(self,cnode,ctype,trans=0.02,load=0.0005,dflag=False):
        '''interpolate timing based on the specified transition and load'''
        lutT = self.get_cell_timing(cnode,ctype=ctype,todf=False)
        pL = []
        for k in lutT:
            lut = lutT[k]
            rpin,pin,cond = k[0].split(',')
            p = self.table_lookup(lut,trans=trans,load=load,dflag=dflag)
            pL += [p]
            if dflag:
                print(f'{rpin}->{pin}@({cond})= {p}\n')
        return np.array(pL).mean()

    def lookup_cell_power(self,cnode,ctype,related_pg_pin='VDD',trans=0.02,load=0.001,dflag=False):
        '''interpolate power based on the specified transition and load'''
        lutP = self.get_cell_power(cnode,ctype=ctype)
        piL,poL = [],[]
        for k in lutP.keys():
            lut = lutP[k]
            rpin,pin,cond = k[0].split(',')
            p = self.table_lookup(lut,trans=trans,load=load,dflag=dflag)
            if rpin=='':
                piL += [p]
            else:
                poL += [p]
            if dflag:
                print(f'[{"IN" if rpin=="" else "OUT"}] {rpin}->{pin}@({cond})= {p}\n')
        pi,po = np.array(piL).mean(),np.array(poL).mean()
        return pi+po,pi,po
    
    '''
    def lookup_cell_power(self,cnode,ctype,related_pg_pin='VDD',trans=0.02,load=0.001,
        onlyClock=False,onlyOutput=False,onlyInput=False,onlyData=False,dflag=False):
        vl = np.array([])
        for pin in cnode['pin'].keys():
            inode = cnode['pin'][pin]
            clock = inode['clock'].upper() if inode.get('clock')!=None else 'false'
            di = inode['direction'].upper() if inode.get('direction')!=None else 'BID'
            if inode.get('internal_power')==None:
                continue
            if onlyInput==True:
                if di[0]!='I':
                    continue
            if onlyOutput==True:
                if di[0]!='O':
                    continue
            if onlyData==True:
                if di[0]=='O' or clock[0]=='T': # clock:true
                    continue
            if onlyClock==True: # grab clock power only
                if clock[0]!='T': # clock:true
                    continue
            tables = inode['internal_power']
            for tnode in tables:
                if tnode.get(ctype)==None:
                    continue
                pg = tnode['related_pg_pin'].strip('"') if tnode.get('related_pg_pin')!=None else ''
                if bool(re.match(pg,related_pg_pin))==False:
                    continue
                rinp = tnode['related_pin'] if tnode.get('related_pin')!=None else ''
                cond = tnode['when'] if tnode.get('when')!=None else ''            
                lut = tnode[ctype]
                vi = self.table_lookup(lut,trans,load,dflag)
                vl = np.append(vl,vi)
                if dflag:
                    print(f'{rinp}->{pin}@({cond})= {vi}')
        avg = vl.mean() if vl.size>0 else 0
        std = vl.std() if vl.size>0 else 0
        chk = std*100/avg if avg>0 else 0
        if chk>=15: # warning threshold
            print(f'WARN! {ctype} variation among {vl.size} lut {chk:.2f}% > 15%' )
        if dflag:
            print(f'mean of {vl.size} lut= {avg}')
        return avg
    '''
    def lookup_cell_slope(self,cnode,dflag=False):
        lutT = self.get_cell_timing(cnode,ctype='transition')
        #lutT = {v:lutT[v] for v in lutT if 'disable' not in str(v)} # waive tri-state disable
        sr,sf = np.array([]),np.array([])
        for k in lutT:
            d = self.lut2df(lutT[k])
            try:
                s = (d.columns[-2]-d.columns[1])/(d.iloc[1:-2,-2]-d.iloc[1:-2,1]).mean()
            except:
                s = np.nan
            if 'rise' in str(k):
                sr = np.append(sr,s) 
            else:
                sf = np.append(sf,s) 
            if dflag:
                print(f'lookup table: {k}, slope={s:.5g}\n{d}\n')
        #print(f'average total {len(sr)} sr, {len(sf)} sf tables')
        return sr.mean(),sf.mean()
    
    def lib2df(self,lnode):
        '''convert liberty DB to pandas dataframe'''
        cnodeL = self.get_cells(lnode,'.*')
        dfL,cellL = [],[]
        for cnode in cnodeL:
            lutL = self.get_cell_timing(cnode)
            if len(lutL)==0:
                continue
            d = pd.DataFrame.from_dict(lutL,orient='index')
            dfL += [d]
            cellL += [cnode['name']]
        df = pd.concat(dfL,keys=cellL)
        df.index.names = ['cell','arc','ttype','ctype']
        return df.sort_index(axis=1)

    # convert lut to dataframe, lut is a dictionary as {'index_1':[...],'index_2':[...],'values':[...]}
    def lut2df(self,lut):
        '''lut is a dictionary as {'y:index_1':[...],'x:index_2':[...],'z:values':[...]}'''
        if len(lut)<=1: # one value
            d = pd.DataFrame(np.array(lut.values()),index=0)
        elif len(lut)==2: # one-dimension
            y,z = map(np.array,lut.values())
            d = pd.DataFrame(z.ravel(),index=y)
            d.index.name = list(lut.keys())[0]
        else: # two-dimension
            lut = {v:lut[v] for v in sorted(lut.keys())}
            y,x,z = map(np.array,lut.values())
            d = pd.DataFrame(z.reshape(len(y),len(x)),columns=x,index=y)
            d.index.name = list(lut.keys())[0]
            d.columns.name = list(lut.keys())[1]
        return d

    # convert lut to the lest-square coefficients (according to the lut dimension)
    def lut2lsCoeff(self,lut,order=2,trans=0.02,load=0.001,dflag=False):
        '''lut is a dictionary as {'y:index_1':[...],'x:index_2':[...],'z:values':[...]}
           return LS coefficient and evaluation based on the specified trans & load'''
        if len(lut)<=1:
            C,p = np.array([0.0]),0
        elif len(lut)<3:
            x,y = map(np.array,lut.values())
            X = np.array([np.ones(y.shape)]+[x**i for i in range(1,order+1)]).T
            C,_,_,_ = scipy.linalg.lstsq(X,y)
            if dflag==True:
                eq =  ['']+[f'x^{i}' for i in range(1,order+1)]
                print(' '.join([f'{c:+.2g}{eq[i]}' for i,c in enumerate(C)]))
                t = np.arange(x[0],x[-1],x[-1]/50)
                T = np.array([np.ones(t.shape)]+[t**i for i in range(1,order+1)]).T
                plt.figure(figsize=(8,5))
                plt.plot(x,y,marker='o',label='original')
                plt.plot(t,np.dot(T,C),alpha=0.5,lw=2,label='predict')
                plt.scatter(trans,np.dot(np.array([1]+[trans**i for i in range(1,order+1)]).T,C),s=20,c='r',marker='s')
                plt.grid()
                plt.show()
            x = trans
            p = np.dot(np.array([1]+[x**i for i in range(1,order+1)]).T,C)
        else:
            y,x,z = map(np.array,lut.values())
            gx,gy = map(np.ravel,np.meshgrid(x,y))
            mx,my = map(np.ravel,np.meshgrid(range(1,order),range(1,order)))
            mxy = list(zip(mx,my))
            X = np.array(
                [np.ones(gx.shape)]+
                [gx**i for i in range(1,order+1)]+
                [gy**i for i in range(1,order+1)]+
                [gx**i * gy**j for i,j in mxy] # covariate (order-1)
                ).T
            C,_,_,_ = scipy.linalg.lstsq(X,z)
            if dflag==True:
                eq =  ['']
                eq += [f'x^{i}' for i in range(1,order+1)]
                eq += [f'y^{i}' for i in range(1,order+1)]
                eq += [f'x^{i}*y^{j}' for i,j in mxy] # covariate (order-1)
                print(' '.join([f'{c:+.2g}{eq[i]}' for i,c in enumerate(C)]))
                tx = np.arange(x[0],x[-1],x[-1]/50)
                ty = np.arange(y[0],y[-1],y[-1]/50)
                px,py = map(np.ravel,np.meshgrid(tx,ty)) # flatten the vectors
                T = np.array([np.ones(px.shape)]+
                    [px**i for i in range(1,order+1)]+
                    [py**i for i in range(1,order+1)]+
                    [px**i * py**j for i,j in mxy]).T
                px,py = np.meshgrid(tx,ty)
                pz = np.dot(T,C).reshape(px.shape)
                gx,gy = np.meshgrid(x,y)
                f = plt.figure()
                f.set_size_inches(8,7)
                ax = Axes3D(f)
                ax.scatter(gx,gy,z.reshape(gx.shape),s=10,label='original')
                ax.plot_surface(px,py,pz,alpha=0.2,edgecolor='w',zorder=0)
                ax.scatter(load,trans,np.dot(np.array([1]+
                    [load**i for i in range(1,order+1)]+
                    [trans**i for i in range(1,order+1)]+
                    [load**i * trans**j for i,j in mxy]).T,C),s=30,c='r',marker='d',zorder=100)
                ax.set_xlabel('index_2')
                ax.set_ylabel('index_1')
                ax.view_init(30,250)
                plt.show()
            y,x = np.asarray(trans),np.asarray(load) # scalar to array
            T = np.array([np.ones(x.shape)]+
                    [x**i for i in range(1,order+1)]+
                    [y**i for i in range(1,order+1)]+
                    [x**i * y**j for i,j in mxy]).T
            p = np.dot(T,C)
        return C,p

    def plot_lut(self,lutT,keys=None,title=None,xylabel=('load','tran')):
        '''lutT := a dictionary encapsulates lut with key values as ('arc','ttpy','ctyp') form
           keys := a key or a list of key values described above 
           wherein lut is a dictionary as {'index_1':[...],'index_2':[...],'values':[...]}'''
        if isinstance(keys,list)==False:
            keys = [keys]
        f = plt.figure()
        f.set_size_inches(8,7)
        ax = Axes3D(f)
        for k in keys:
            if len(k)==3: # unpack timing
                arc,ttype,ctype = k
            else: # unpack power
                arc,ctype = k
            lut = lutT[k]
            lut = {v:lut[v] for v in sorted(lut.keys())}
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
        ax.set_xlabel(f'index_2:{xylabel[0]}') # load, clock
        ax.set_ylabel(f'index_1:{xylabel[1]}') # tran, data
        ax.set_zlabel('values')
        plt.legend()
        plt.show()
    
    def plot_cell_timing(self,cnode,arc,ctype,xylabel=('load','trans')):
        lutT = self.get_cell_timing(cnode,ctype=ctype)
        keyL = [(a,t,c) for a,t,c in lutT if ctype in c and arc==a]
        self.plot_lut(lutT,keys=keyL)
    
    def plot_cell_constraint(self,cnode,arc=None,ctype=None,xylabel=('clock','data'),shape=(9,7)):
        lutT = self.get_cell_timing(cnode,ctype=ctype)
        luts = lutT.get(tuple([arc+',']+['setup_rising',ctype]))
        luth = lutT.get(tuple([arc+',']+['hold_rising', ctype]))
        if luts==None or luth==None:
            print('error: invalidate key value')
            return
        cell = cnode['name']
        y,x,ts = map(np.array,luts.values())
        y,x,th = map(np.array,luth.values())
        f = plt.figure()
        f.set_size_inches(8,7)
        f.subplots_adjust(hspace=0.05,wspace=0.05,top=0.95,right=0.7)
        ax = Axes3D(f)
        gx,gy = np.meshgrid(x,y)
        px = np.arange(0,x[-1],(x[-1]/30)) # col
        py = np.arange(0,y[-1],(y[-1]/30)) # row
        gpx,gpy = np.meshgrid(px,py)
        pzs = interpolate.interp2d(x,y,ts.reshape(len(y),len(x)),kind='linear')(px,py)
        pzh = interpolate.interp2d(x,y,th.reshape(len(y),len(x)),kind='linear')(px,py)
        ax.plot_surface(gpx,gpy,pzs, # setup
            antialiased=False,linewidth=0,edgecolors='none',
            alpha=0.8,cmap=plt.cm.Blues)
        ax.plot_surface(gpx,gpy,-pzh, # hold
            antialiased=False,linewidth=0,edgecolors='none',
            alpha=0.8,cmap=plt.cm.Reds)
        ax.plot_wireframe(gpx,gpy,np.zeros(gpx.shape),lw=0.9,color='gray')
        ax.set_title(f'{cell}\n{arc},{ctype}')
        ax.set_xlabel(xylabel[0]) # index_2: load, clock
        ax.set_ylabel(xylabel[1]) # index_1: tran, data
        ax.view_init(30,250)
        l1 = plt.matplotlib.lines.Line2D([0],[0],c='gray',marker ='o')
        l2 = plt.matplotlib.lines.Line2D([0],[0],c='blue',marker='o')
        l3 = plt.matplotlib.lines.Line2D([0],[0],c='red',marker='o')
        ax.legend([l1,l2,l3], ['clock','setup','hold'],numpoints=1)
        plt.show()
    
    # metric extraction
    def cellLutCoeff(self,cnode,trans=0.04,load=0.0017,todf=True):
        '''convert lut to LS coefficient in {arc:{metric:..,coeff:..}, ..} fashion,
           encapsulated as a dataframe if todf==True, otherwise a dict will be returned'''
        cmL = {}
        lutT = self.get_cell_timing(cnode) # coefficient of timing & constraint 
        for k in lutT.keys():
            lut = lutT[k]
            y = trans # index_1: trans | data 
            x = y/2 if 'constraint' in str(k) else load # 1/2 data as clock 2022/08
            #C,p = self.lut2lsCoeff(lut,trans=trans,load=load)
            C,p = self.lut2lsCoeff(lut,trans=y,load=x)
            cmL[k] = [p,list(C)]
        lutP = self.get_cell_power(cnode) # coefficient of power 
        for k in lutP.keys():
            lut = lutP[k]
            C,p = self.lut2lsCoeff(lut,trans=trans,load=load)
            cmL[k] = [p,list(C)]
        dc = pd.DataFrame(cmL,index=['metric','coeff']).T
        dc.index.name = 'arc'
        if todf==False: # to dict stream (JSON)
            nan = None
            dc = dc.to_dict('index') 
        return dc
    
    def cellMetric(self,cnode,trans=0.04,load=0.0017):
        '''extract cell metric form cell node in dict (from CCS or JSON),
           encapsulated lut efficient in {arc:{metric:..,coeff:..}..} fashion'''
        fp = cnode['cell_footprint'] if cnode.get('cell_footprint')!=None else 'nan'
        area = float(cnode['area']) if cnode.get('area')!=None else 0.0
        leak = float(cnode['cell_leakage_power']) if cnode.get('cell_leakage_power')!=None else self.lookup_cell_leakage(cnode)
        cap = self.lookup_cell_pincap(cnode)
        sr,sf = self.lookup_cell_slope(cnode)
        # encapsulated LS coefficient, in {arc:{metric:..,coeff:..}..} fashion 
        dc = self.cellLutCoeff(cnode,trans,load,todf=False)
        tr = np.array([dc[k]['metric'] for k in dc if 'rise_transition' in k]).mean()
        tf = np.array([dc[k]['metric'] for k in dc if 'fall_transition' in k]).mean()
        cr = np.array([dc[k]['metric'] for k in dc if 'cell_rise' in k]).mean()
        cf = np.array([dc[k]['metric'] for k in dc if 'cell_fall' in k]).mean()
        pr = np.array([dc[k]['metric'] for k in dc if 'rise_power' in k]).mean()
        pf = np.array([dc[k]['metric'] for k in dc if 'fall_power' in k]).mean()
        return cnode['name'],fp,area,leak,cap,sr,sf,tr,tf,cr,cf,pr,pf,dc

    def dump_libmetric(self,lnode,trans=0.04,load=0.0017,cname_re='.*',out=None,aspkl=False):
        '''extract library metric as DataFrame with columns=
           ['fp', 'area', 'leak', 'cap', 'sr', 'sf', 'tr', 'tf', 'dr', 'df', 'pr', 'pf', 'metrics'],
           wherein the metrics is encapsulated as {arc:{'metric':..,'coeff':..}, ..}'''
        cmL = []
        cnodeL = self.get_cells(lnode,cname_re=cname_re)
        for cnode in cnodeL:
            cm = self.cellMetric(cnode,trans,load)
            cmL += [cm]
        dm = pd.DataFrame(cmL,columns=['cell','fp','area','leak','cap','sr','sf','tr','tf','dr','df','pr','pf','metrics'])
        if out!=None:
            if aspkl==False: # CSV
                dm.to_csv(out,index=False,float_format='%g')
            else: # Pickle
                dm.to_pickle(out)
            print(f'liberty metric was saved into {out} ...')
        return dm.set_index(['cell'])
    
    def dump_liblut(self,lnode,trans=0.04,load=0.0017,cname_re='.*',out=None,aspkl=False):
        '''encapsulate cell timing & power tables described in pin groups into {arc:lut} fashion'''
        cmL = []
        cnodeL = self.get_cells(lnode,cname_re=cname_re)
        for cnode in cnodeL:
            cell,fp,area,leak,cap,sr,sf,tr,tf,cr,cf,pr,pf,_ = self.cellMetric(cnode,trans,load)
            lutT = self.get_cell_timing(cnode) # timing & constraint lut
            lutP = self.get_cell_power(cnode)  # power lut
            cmL += [[cell,fp,area,leak,cap,sr,sf,tr,tf,cr,cf,pr,pf,lutT,lutP]]
        dm = pd.DataFrame(cmL,columns=['cell','fp','area','leak','cap','sr','sf','tr','tf','cr','cf','pr','pf','lutT','lutP'])
        if out!=None:
            if aspkl==False:
                dm.to_csv(out,index=False,float_format='%g')
            else:
                dm.to_pickle(out)
            print(f'dump liberty into lut {out} ...')
        return dm
    
    def dump_json(self,libnode,out,cname_re=None,indent=None,sort_keys=False):
        lnode = libnode.copy()
        if cname_re != None:
            cnodeL = self.get_cells(lnode,cname_re=cname_re)
            lnode['cell'] = {v['name']:v for v in cnodeL}
        jnode = json.dumps(lnode,sort_keys=sort_keys,ensure_ascii=False,indent=indent)
        with open(out,mode='w') as f:
            f.write(jnode)
        print(f'liberty object is saved into {out} (JSON) ...')

class LibertyMetric(Liberty):
    def __init__(self):
        super().__init__()
        print('init class LibertyMetric ...')

    def metric2df(self,m):
        '''convert the metric in JSON (dictionary) into a dataframe'''
        nan = np.nan
        if type(m)==str: # convert string to JSON (dictionary)
            m = eval(m) 
        d = pd.DataFrame(m.values(),index=m.keys())
        return d
    
    def get_cell_metric(self,dml,cell,todf=True):
        '''encapsulate cell metric into a dataframe {arc:metric,[coeff]}
           dml := corner liberty metric'''
        dm = dml.loc[cell]['metrics']
        if type(dm)==str: # convert to JSON
            dm = eval(dm) 
        if todf==True: # convert to DataFrame
            dm = self.metric2df(dm)
        return dm   

    def lscFitting(self,C,index_1,index_2=0,dflag=False):
        '''calculate timing from LS coefficient'''
        y,x = index_1,index_2
        if len(C)==3: # dimension-1
            C = [C[0],0,0,C[1],C[2],0] # pad zeros for x terms
            #p = np.dot(np.array([1,y,y**2]).T,C)
            #C = np.pad(C,(0,6-len(C))) # pad zero
            p = np.dot(np.array([np.ones_like(x),x,x**2,y,y**2,x*y]).T,C)
        # debug LSC regression
        if dflag:
            ty,tx = np.linspace(y*0.5,y*1.5,20),np.linspace(x*0.5,x*1.5,20)
            gx,gy = np.meshgrid(tx,ty)
            px,py = gx.ravel(),gy.ravel()
            pz = np.dot(np.array([np.ones_like(px),px,px**2,py,py**2,px*py]).T,C)
            pz = pz.reshape(gx.shape)
            f = plt.figure()
            f.set_size_inches(8,7)
            ax = Axes3D(f)
            ax.plot_surface(gx,gy,pz,alpha=0.2,edgecolor='w',zorder=0)
            ax.scatter(x,y,p,s=50,color='r',label='querry')
            ax.text(x,y,p,s=f'{p:.3f} @{y:.3f},{x:.3f}')
            ax.set_xlabel('index_2')
            ax.set_ylabel('index_1')
            plt.show()
        return p
    
    def cellLSCSurface(self,dmc,keys,x=np.linspace(0,0.01,20),y=np.linspace(0,0.06,20)):
        ''' reconstruct timing surface from LS coefficient, where dmc (in JSON) encapsulates cell metrics by timing arc;
            x,y represent the axis range of index_2,index_1 respectively'''
        if isinstance(dmc,pd.DataFrame): # convert Dataframe to JSON
            dmc = dmc.to_dict(orient='index')
        keys = [keys] if isinstance(keys,list)==False else keys
        gx,gy = np.meshgrid(x,y)
        px,py = map(np.ravel,(gx,gy))
        T = np.array([np.ones(px.shape),px,px**2,py,py**2,px*py]).T # LS(order=2) grid
        f = plt.figure(figsize=(8,7))
        f.set_size_inches(8,7)
        ax = Axes3D(f)
        for k in keys:
            C = dmc[k]['coeff']
            pz = np.dot(T,C).reshape(gx.shape)
            if 'hold' in str(k):
                pz = -pz
            ax.scatter(gx,gy,pz,alpha=0.5,s=5,label=k)
            ax.plot_surface(gx,gy,pz,alpha=0.3)
        ax.set_xlabel('x: index_2') # load,clock
        ax.set_ylabel('y: index_1') # tran,data
        ax.set_zlabel('values')
        ax.view_init(30,250)
        plt.legend()
        plt.show()
        
    def dataAugmentation(self,df):
        df['ta'] = df[['tr','tf']].mean(axis=1) # average cell transition
        df['da'] = df[['dr','df']].mean(axis=1) # average cell transition
        df['sa'] = df[['sr','sf']].mean(axis=1) # average cell driving
        df['pa'] = df[['pr','pf']].mean(axis=1) # average cell power
        df['dps'] = df['da']/df['sa'] # delay per-slope
        df['tps'] = df['ta']/df['sa'] # transition per-slope
        df['pps'] = df['pa']/df['sa'] # power per-slope
        itemL = df.columns.drop('metrics').tolist()+['metrics']
        return df[itemL]

#liberty = Liberty()
liberty = LibertyMetric()
