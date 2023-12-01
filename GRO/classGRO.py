'''
Grid RO (GRO) Compiler v0.1 build 2022/6

Copyright (C) 2022, by DipSci Technology Corpopation, LTD.
This software tool is independently developed and owned by DipSci Technology Co., Ltd.
It is freely licensed to industry and academia to use, share, reproduce, or improve, provided 
this copyright notice is retained or the initial creater is listed on the contributors list.
Derivatives developed for profit based on this software tool must still include this copyright 
notice or credit the initial creater on the contributors list.

Contributors:
hockchen, paul, michael - initial creater

Revision history:
2022/05/17 hockchen Init
2022/05/25 paul     Fixed the problem that RO could not oscillate
2022/05/26 hockchen Enhanced RO delay calculation
2022/05/30 hockchen Command argument parser
2022/06/08 paul     SPICE deck generation
2022/12/06 hockchen SPI wrapper for GRO and SPI controller
'''
import numpy as np
import os, re
#pip install libertymetric
#from libertymetric.classLiberty import LibertyMetric
#pip install DTCO
from libMetric import LibertyMetric
from datetime import datetime

class ROCompiler(LibertyMetric):
    def __init__(self): # init with the caller source directory
        super().__init__()
        self.src = os.path.dirname(os.path.realpath(__file__))
        self.cfg = {}
        self.libDB = {}   # liberty DB in JSON format
        self.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.status = None
        print(f'init class GRO Compiler {self.date} ...')
        print(f'program source directory: {self.src}')

    def loadConfig(self,cfg):
        self.cfg['configFile'] = cfg
        print(f'parse configuration {cfg} ...')
        with open(cfg,'r') as fh:
            text = fh.read()
        text = '\n'.join([f'{v.strip(";")};' if ('{' not in v and '=' in v) else v for v in text.split('\n') if v.strip()!='']) # patch single assignment with ;
        text = re.sub(r'#.*\n|//.*\n|\\\n|[ \t\n]+','',text) # remove white space
        text = re.sub(r'/\*.*\*/','',text) # remove multi comments
        text = re.sub(r'#.*|//.*','',text) # remove single comment
        for item in re.split('[};]+',text):
            #print(f'=>{item}')
            if item=='':
                continue
            token,v = re.split('[={]+',item)
            if token in ['top','outPath','targetCond'] :
                self.cfg[token] = v.strip("\"'")
            else:
                self.cfg[token] = eval('{'+v+'}') #eval(f'{{{v}}}')
        # legalize config
        self.cfg['top'] = self.cfg.get('top','GRO_TOP') # disable configuration for SPI wrapper 2022/12
        if isinstance(self.cfg.get('lib'),set): # enclose set to list with a 'default' key
            self.cfg['lib'] = {'default':list(self.cfg.get('lib',{}))} # to dictionary
        if isinstance(self.cfg.get('lpe'),set): # enclose set to list with a 'default' key
            self.cfg['lpe'] = list(self.cfg.get('lpe',{})) # to dictionary
        return self.cfg
    
    def loadLib(self,file):
        '''read CCS or JSON format by extension'''
        tag = file.split('.')[-1]
        if tag=='json':
            lnode = self.load_json(file)
        elif tag=='lib':
            lnode = self.read_lib(file)
        else: # metric CSV
            lnode = None # currently, not support
            #lnode = self.load_metric(file)
        #self.lnode = lnode
        return lnode
    
    # query GRO components
    def queryCellWithFP(self,lnode,fp='.*'):
        '''lnode := liberty node in JSON'''
        cellL = []
        for cell in lnode['cell']:
            cnode = lnode['cell'][cell]
            if cnode.get('cell_footprint')!=None and re.match(fp,cnode['cell_footprint']):
                cellL += [cell]
                print(f'{cnode["cell_footprint"]}: {cell}')
        return cellL
    
    def queryCellFromLibDB(self,regexp,target=None):
        retL = []
        for cond in self.libDB:
            if target!=None and cond!=target:
                break
            for lnode in self.libDB[cond]:
                for cell in lnode['cell']:
                    if re.match(regexp,cell)!=None:
                        cnode = lnode['cell'][cell]
                        pinL = list(cnode['pin'].keys())
                        retL += [(cond,lnode['library'],cell,pinL)]
        return retL # list of (condition,library,cell,pins)

    def getCellFromLibDB(self,cell,target):
        target = target or self.cfg.get('targetCond','TT')
        ret = self.queryCellFromLibDB(cell,target)
        if ret==[]:
            print(f'ERROR! cannot find {cell}@{target} in JSON DB')
            return None,None
        cond,lib,cell,pins = ret[0]
        lnode = [v for v in self.libDB[target] if v['library']==lib][0]
        cnode = lnode['cell'][cell]
        return lnode,cnode

    # liberty inetgrity check
    def integrityCheck(self,target):
        if self.libDB=={}:
            print('ERROR! please commit liberty JSON first')
            return False
        if target not in self.cfg.get('lib',{}).keys():
            print(f'ERROR! target condition should be one of {list(self.cfg.get("lib",{}).keys())}')
            return False
        for term in ['enGate','invGate','cntDFF','delayLine']:
            if self.cfg.get(term)==None:
                print(f'ERROR! require {term} in the configuration')
                return False
        
        print(f'INFO: integrity check, target ocnd: {target} ...')
        # enable gate
        lnode,cnode = self.getCellFromLibDB(self.cfg['enGate']['name'],target)
        if cnode!=None:
            iPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='input']
            oPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='output']
            cap = self.lookup_cell_pincap(cnode) # cap
            self.cfg['enGate']['en'] = self.cfg['enGate'].get('en',iPin[1])
            self.cfg['enGate']['in'] = self.cfg['enGate'].get('in',iPin[0])
            self.cfg['enGate']['out'] = self.cfg['enGate'].get('out',oPin[0])
            self.cfg['enGate']['cap'] = self.cfg['enGate'].get('cap',cap)
            print(f'{cnode["name"]}: I:{iPin}, O:{oPin}')

        # INV gate
        lnode,cnode = self.getCellFromLibDB(self.cfg['invGate']['name'],target)
        if cnode!=None:
            iPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='input']
            oPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='output']
            cap = self.lookup_cell_pincap(cnode) # cap
            self.cfg['invGate']['in'] = self.cfg['invGate'].get('in',iPin[0])
            self.cfg['invGate']['out'] = self.cfg['invGate'].get('out',oPin[0])
            self.cfg['invGate']['cap'] = self.cfg['invGate'].get('cap',cap)
            print(f'{cnode["name"]}: I:{iPin}, O:{oPin}')
        
        # DFF
        lnode,cnode = self.getCellFromLibDB(self.cfg['cntDFF']['name'],target)
        if cnode!=None:
            ckPin = [v for v in cnode['pin'] if cnode['pin'][v].get('clock')][0]
            iPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='input' and v != ckPin]
            oPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='output']
            reset = self.cfg['cntDFF'].get('clear',iPin[0]) # reset
            self.cfg['cntDFF']['clock'] = ckPin
            self.cfg['cntDFF']['clear'] = reset
            self.cfg['cntDFF']['in'] = [v for v in iPin if v != reset][0]
            self.cfg['cntDFF']['out'] = self.cfg['cntDFF'].get('out',oPin[0]) # Q or QN
            print(f'{cnode["name"]}: CK:{ckPin}, Reset:{self.cfg["cntDFF"]["clear"]}, O:{oPin}')
        
        # Delay-line & update in/out port decration
        for dl in self.cfg['delayLine']:
            cell,period,*_ = self.cfg['delayLine'][dl] # drop previous I/O setting
            lnode,cnode = self.getCellFromLibDB(cell,target)
            if cnode!=None:
                iPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='input']
                oPin = [v for v in cnode['pin'] if cnode['pin'][v]['direction']=='output']
                self.cfg['delayLine'][dl] = [cell,period,iPin,oPin] #+= [iPin,oPin]
                print(f'{dl},{period}ps, {cell}: I:{iPin}, O:{oPin}')
        return self.cfg
    
    # grab transition, load and delay of the specified base gate
    def baseGateInfo(self,cell=None,trans=0.05,load=None,target=None):
        '''use self load if load is not specified'''
        target = target or self.cfg.get('targetCond','TT') 
        cell = cell or self.cfg.get('enGate',{}).get('name','ND2D1')
        #cnode = self.get_cell(lnode,cell)
        lnode,cnode = self.getCellFromLibDB(cell,target)
        cap = self.lookup_cell_pincap(cnode)
        pins = len(self.get_pins(cnode,direction='in').keys())
        FO = load if load!=None else cap*pins*1.0 # depands on pin#, 20% wire-load margin?
        ta = trans
        for ii in range(5): 
            tr = self.lookup_cell_timing(cnode,'rise_transition',trans=ta,load=FO)
            tf = self.lookup_cell_timing(cnode,'fall_transition',trans=ta,load=FO)
            dr = self.lookup_cell_timing(cnode,'cell_rise',trans=ta,load=FO)
            df = self.lookup_cell_timing(cnode,'cell_fall',trans=ta,load=FO)
            ta = (tr+tf)/2 # converged transition
            da = (dr+df)/2 # converged delay
            #print(f'{ii} trans={ta:.5f}')
        print(f'converged transition & load of {lnode["library"]}/{cell}: {ta:.4f}, {FO:.4f}, delay:{da:.4f}')
        return ta,FO,da

    def genDLModule(self,dl,fh):
        cell,period,iPin,oPin = self.cfg['delayLine'][dl]
        tran,load,delay = self.baseGateInfo(cell,trans=0.05) # delay-cell, self load
        cap = self.cfg['enGate']['cap']+self.cfg['invGate']['cap']
        _,_,da = self.baseGateInfo(self.cfg['enGate']['name'],tran,cap) # enable & split gate
        delay = np.ceil(delay*1000) # delay-line delay (ps)
        da = np.ceil(da*1000) # enable delay (ps)
        stage = (period/2-da)/delay
        stage = int(round(stage/2)*2)
        print(f'{dl},{period}, {cell}: delay={delay:.2f}ps, stage={stage}')
        fh.write(f'// {dl},{period}ps, {cell}: I:{iPin}, O:{oPin}, delay={delay:.2f}ps, stage:{stage}\n')
        fh.write(f'module {dl}_{cell} ( .EN(EN), .RO(RO) );\n')
        fh.write('input EN;\n')
        fh.write('output RO;\n')
        fh.write(f'wire [{stage+2}:0] NET;\n')
        fh.write(f'{self.cfg["enGate"]["name"]} u_enable ( // {da}ps\n')
        fh.write(f'    .{self.cfg["enGate"]["en"]}(EN),\n')
        fh.write(f'    .{self.cfg["enGate"]["in"]}(NET[{stage+1}]),\n')
        fh.write(f'    .{self.cfg["enGate"]["out"]}(NET[1]) );\n')
        fh.write(f'{self.cfg["invGate"]["name"]} u_inv (\n')
        fh.write(f'    .{self.cfg["invGate"]["in"]}(NET[{stage+1}]),\n')
        fh.write(f'    .{self.cfg["invGate"]["out"]}(RO) );\n')
        cumd = da
        for ii in range(1,stage+1):
            cumd += delay
            icon = ', '.join([f'.{v}(NET[{ii}])' for v in iPin])
            ocon = ', '.join([f'.{v}(NET[{ii+1}])' for v in oPin])
            fh.write(f'{cell} u_u{ii} ( {icon}, {ocon} ); // {cumd:.2f}ps\n')
        fh.write('endmodule\n\n')

    def genSDLI(self,fh):
        dlNum = len(self.cfg['delayLine'])
        bitNum = int(np.ceil(np.log2(dlNum)))
        busNum = 2**bitNum
        fh.write('module SDLI(EN, SEL, ENO);\n')
        fh.write('output reg [%d:0] ENO;\n' % (busNum-1))
        fh.write('input [%d:0] SEL;\n' % (bitNum-1))
        fh.write('input EN;\n')
        fh.write('always @(*) begin\n')
        fh.write('    if (EN) begin\n')
        fh.write('        case(SEL)\n')
        for ii in range(busNum):
            fh.write('            %d: ENO = %d\'b%s;\n' % (ii,busNum,bin(2**ii)[2:].zfill(busNum)))
        fh.write('            default: ENO = %d\'b%s;\n' % (busNum,bin(0)[2:].zfill(busNum)))
        fh.write('        endcase\n')
        fh.write('    end else begin\n')
        fh.write('        ENO = %d\'b%s;\n' % (busNum,bin(0)[2:].zfill(busNum)))
        fh.write('    end\n')
        fh.write('end\n')
        fh.write('endmodule\n\n')

    def genSDLO(self,fh):
        dlNum = len(self.cfg['delayLine'])
        bitNum = int(np.ceil(np.log2(dlNum)))
        busNum = 2**bitNum
        fh.write('module SDLO (EN, SEL, ENO);\n')
        fh.write('output reg ENO;\n')
        fh.write('input [%d:0] EN;\n' % (busNum-1))
        fh.write('input [%d:0] SEL;\n'% (bitNum-1))
        fh.write('always @ (*) begin\n')
        fh.write('case (SEL)\n')
        for ii in range(busNum):
            fh.write('    %d\'b%s : ENO = EN[%d];\n' % (bitNum,bin(ii)[2:].zfill(bitNum),ii))
        fh.write('    default : ENO = EN[0];\n')
        fh.write('    endcase\n')
        fh.write('end\n')
        fh.write('endmodule\n\n')

    def genRippleCounter(self,fh):
        fh.write('module RippleCounter(CLK, COUNT, RSTN);\n')
        fh.write('input CLK, RSTN;\n')
        fh.write(f'output [{self.cfg["cntDFF"]["bits"]-1}:0] COUNT;\n')
        for ii in range(self.cfg['cntDFF']['bits']):
            net = 'CLK' if ii==0 else f'n_inv{ii-1}_o'
            fh.write(f'{self.cfg["cntDFF"]["name"]} u_dff{ii} (\n')
            fh.write(f'    .{self.cfg["cntDFF"]["out"]}(COUNT[{ii}]),\n')
            fh.write(f'    .{self.cfg["cntDFF"]["clock"]}({net}), .{self.cfg["cntDFF"]["in"]}(n_inv{ii}_o), .{self.cfg["cntDFF"]["clear"]}(RSTN));\n')
            fh.write(f'{self.cfg["invGate"]["name"]} u_inv{ii} (.{self.cfg["invGate"]["out"]}(n_inv{ii}_o), .{self.cfg["invGate"]["in"]}(COUNT[{ii}]));\n')
        fh.write('endmodule\n\n')

    def genSPICESim(self):
        if os.path.exists(self.cfg['outPath']+'/spice')==False:
            print(f"ERROR! output path {self.cfg['outPath']}/spice does not exist")
            return False
        target = self.cfg.get('targetCond','TT')
        key = list(self.libDB[target][0]['operating_conditions'].keys())[0] # use the 1st opc
        opc = self.libDB[target][0]['operating_conditions'][key]

        out = f"{self.cfg['outPath']}/spice/sim_GRO.sp"
        ofh = open(out,'w') # output verilog file
        print("*********Start Testing***********",file=ofh)
        print(f".param SUPPLY={opc['voltage']}",file=ofh)
        print(f".param TEMP={opc['temperature']}",file=ofh)
        print(".temp TEMP",file=ofh)
        print(".inc './merge.spx'",file=ofh)
        print(f".inc './{self.cfg['top']}.spi'",file=ofh)
        for card in self.cfg.get('model_card',{}):
            model =  self.cfg.get('model_card',{}).get(card,'undefine')
            print(f".lib '{model}' {card}",file=ofh)
        print(".probe tran V(RO)",file=ofh)
        print(".option probe=1",file=ofh)
        print(".option brief post",file=ofh)
        print("\n*---Module Instantiate---*",file=ofh)
        print("XX0 EN RSTN",file=ofh)
        dlNum = len(self.cfg['delayLine'])
        selNum = int(np.ceil(np.log2(dlNum)))
        cntNum = self.cfg['cntDFF']['bits']
        sel = ' '.join([f'SEL[{v}]' for v in range(selNum-1,-1,-1)])
        cnt = ' '.join([f'COUNT[{v}]' for v in range(cntNum-1,-1,-1)])
        print(f"+ {sel}",file=ofh)
        print(f"+ {cnt}",file=ofh)
        print(f"+ RO VDD VSS VBP VBN {self.cfg['top']}",file=ofh)
        print("\n*--------Source----------*",file=ofh)
        print("VSUP  VDD VSS SUPPLY",file=ofh)
        print("VBP   VBP VSS SUPPLY",file=ofh)
        print("VVSS  VSS 0 0",file=ofh)
        print("VBN   VBN 0 0",file=ofh)
        print("VEN   EN VSS PWL (0 0 14.9n 0 15n SUPPLY)",file=ofh)
        print("VRSTN RSTN VSS PWL (0 SUPPLY 4.9n SUPPLY 5.0n 0 14.9n 0 15n SUPPLY)",file=ofh)
        # init
        for i in range(selNum):
            print(f"VSEL{i} SEL[{i}] VSS 0",file=ofh)
        print("\n*------Test Setting-----*",file=ofh)
        print(".tran 10p 30n",file=ofh)
        print(".measure tran RO_PERIOD trig V(RO)='0.5*SUPPLY' rise=2 targ V(RO)='0.5*SUPPLY' rise=3 ",file=ofh)
        # test selection
        for code in [f'{v:016b}' for v in range(1,dlNum)]:
            print("\n.alter",file=ofh)
            code = code[::-1] # reverse MSB
            for i in range(selNum):
                tag = 'SUPPLY' if code[i]=='1' else '0'
                print(f"VSEL{i} SEL[{i}] VSS {tag}",file=ofh)
        
        print("\n.end\n",file=ofh)
        print(f'INFO: SPICE testbench was saved into {out} ...')
        ofh.close()

    def compileGRO(self):
        if self.libDB=={}: #or self.lnode==None:
            print('ERROR! please setup liberty JSON first')
            return False
        if os.path.exists(self.cfg['outPath']+'/verilog')==False:
            print(f"ERROR! output path {self.cfg['outPath']}/verilog does not exist")
            return False 
        out = f"{self.cfg['outPath']}/verilog/GRO.v"
        ofh = open(out,'w') # output verilog file
        print(f'// auto build by GRO compiler {self.date}',file=ofh)
        self.genSDLI(ofh)
        self.genSDLO(ofh)
        self.genRippleCounter(ofh)
        for dl in self.cfg['delayLine']: # gen DL Module
            self.genDLModule(dl,ofh)
        # RO top
        dlNum = len(self.cfg['delayLine'])
        bitNum = int(np.ceil(np.log2(dlNum)))
        busNum = 2**bitNum
        print(f'module {self.cfg["top"]} (EN, RSTN, SEL, COUNT, RO);',file=ofh)
        print('input EN;',file=ofh)
        print('input RSTN;',file=ofh)
        print(f'input [{bitNum-1}:0] SEL;',file=ofh)
        print(f'output [{self.cfg["cntDFF"]["bits"]-1}:0] COUNT;',file=ofh)
        print('output RO;',file=ofh)
        print(f'wire [{busNum-1}:0] n_sdli_o;',file=ofh)
        print(f'wire [{busNum-1}:0] n_sdlo_i;',file=ofh)
        print('SDLI u_sdli (.EN(EN), .SEL(SEL), .ENO(n_sdli_o));',file=ofh)
        print('SDLO u_sdlo (.EN(n_sdlo_i), .SEL(SEL), .ENO(RO));',file=ofh)
        print('RippleCounter u_count (.CLK(RO), .COUNT(COUNT), .RSTN(RSTN));',file=ofh)
        # DL instance
        for ii,dl in enumerate(self.cfg['delayLine']):
            cell,period,iPin,oPin = self.cfg['delayLine'][dl]
            print(f'{dl}_{cell} u_{dl} (.RO(n_sdlo_i[{ii}]), .EN(n_sdli_o[{ii}]));',file=ofh)
        for i in range(busNum):
            if (i >= len(self.cfg['delayLine'])):
                print(f'assign n_sdlo_i[{i}] = 0;', file=ofh)
        print('endmodule\n',file=ofh)
        ofh.close()
        print(f'Grid RO module was saved into {out} ...')
        return True
    
    # NOTE! envSetp** functions could be performed without integrity check, but gen** needs 
    # sub flow environments
    def envSetupDC(self):
        if self.libDB=={}:
            print('ERROR! please setup liberty JSON first')
            return False
        if os.path.exists(self.cfg['outPath']+'/synthesis')==False:
            print(f"ERROR! output path {self.cfg['outPath']}/synthesis does not exist")
            return False  
        # check link library
        for db in self.cfg.get('db',{}):
            if os.path.exists(db)==False:
                print(f'ERROR! the specified db {db} doesnot exist')
                return False
        
        print('INFO: setup synthesis environment ...')
        target = self.cfg.get('targetCond','TT')
        opc = list(self.libDB[target][0]['operating_conditions'].keys())[0] # use the 1st library
        dbL = " ".join([v for v in self.cfg.get('db',{})])
        vlog = " ".join(self.cfg.get('vlog',{}))
        dbPath = " ".join([os.path.dirname(v) for v in self.cfg.get('db',{})])
        minp = min([p for c,p,*a in self.cfg['delayLine'].values()])/1000 # min period
        cload = self.cfg['enGate']['cap']*4 # FO4
        out = f"{self.cfg['outPath']}/verilog/config.tcl"
        #top = 'gro_spi' if self.cfg.get('SPI')!=None else 'GRO_TOP'
        top = 'GRO_TOP'
        ofh = open(out,'w') # output configuration Tcl for DC synthesis
        print(f"# auto build by GRO compiler {self.date}",file=ofh)
        print("namespace eval groCfg {",file=ofh)
        print("    array set DC {",file=ofh)
        print(f"        libPath      \"{dbPath}\"",file=ofh)
        print(f"        libName      \"{dbL}\"",file=ofh)
        print(f"        topModule    \"{top}\"",file=ofh)
        print(f"        opConditions \"{opc}\"",file=ofh)
        print(f"        clockPeriod  {minp:.3f}",file=ofh)
        print(f"        drivingCell  \"{self.cfg['enGate']['name']}\"",file=ofh)
        print(f"        drivingPin   \"{self.cfg['enGate']['out']}\"",file=ofh)
        print(f"        vlog         \"{vlog}\"",file=ofh)
        print(f"        cload        {cload:.4f}",file=ofh)
        print("    }",file=ofh)
        print("    proc joinExt {list s} {",file=ofh)
        print("        set result {}",file=ofh)
        print("        foreach x $list {",file=ofh)
        print("            append result \" $x$s\"",file=ofh)
        print("        }",file=ofh)
        print("        return [string trim $result]",file=ofh)
        print("    }",file=ofh)
        print("}",file=ofh)
        print(f'DC configuration Tcl was saved into {out} ...')
        ofh.close()
        return True
    
    def envSetupVsim(self):
        if self.libDB=={}:
            print('ERROR! please setup liberty JSON first')
            return False
        if os.path.exists(self.cfg['outPath']+'/verilog')==False:
            print(f"ERROR! output path {self.cfg['outPath']}/verilog does not exist")
            return False
        
        # check link library
        for vlog in self.cfg.get('vlog',{}):
            if os.path.exists(vlog)==False:
                print(f'ERROR! the specified vlog {vlog} doesnot exist')
                return False
        
        print('INFO: setup gate-sim environment ...')
        dlNum = len(self.cfg['delayLine'])
        selBit = int(np.ceil(np.log2(dlNum)))
        cntBit = self.cfg['cntDFF']['bits']
        #top = self.cfg['top']
        # output configuration Tcl for DC synthesis
        out = f"{self.cfg['outPath']}/verilog/define.vh"
        ofh = open(out,'w') 
        print(f'// auto build by GRO compiler {self.date}',file=ofh)
        print(f'`define {"DL_NUM":8s} {dlNum:5d} // GRO : delay line number',file=ofh)
        print(f'`define {"SEL_BITS":8s} {selBit:5d} // GRO : select bitwidth',file=ofh)
        print(f'`define {"RPC_BITS":8s} {cntBit:5d} // GRO : ripple counter bitWidth',file=ofh)
        #print(f'`define SDFFILE "../synthesis/Netlist/{top}_syn.sdf"',file=ofh)
        spi_cycle  = self.cfg.get("SPI",{}).get('SPI_CYCLE',  3333.3)
        spi_dwidth = self.cfg.get("SPI",{}).get('SPI_DWIDTH', 16)
        spi_intc   = self.cfg.get("SPI",{}).get('INT_CYCLE',  1000)
        spi_pattern = self.cfg.get("SPI",{}).get('PAT_NUM',   10)
        spi_groc   = self.cfg.get("SPI",{}).get('GRO_CYCLE',  10)
        spi_ro_num = self.cfg.get("SPI",{}).get('RO_GRID_NUM', 16)
        spi_ro_enc = self.cfg.get("SPI",{}).get('RO_EN_COUNT_BITS', 8)
        print(f'`define {"RO_ACTIVE":16s} {1:5d} // SPI : ro active flag',file=ofh)
        print(f'`define {"SPI_CYCLE":16s} {spi_cycle:5.1f} // SPI : controller clock period',file=ofh)
        print(f'`define {"SPI_DWIDTH":16s} {spi_dwidth:5d} // SPI : bitWidth',file=ofh)
        print(f'`define {"INT_CYCLE":16s} {spi_intc:5.1f} // SPI : internal clock period',file=ofh)
        print(f'`define {"PAT_NUM":16s} {spi_pattern:5d} // SPI : number of patterns',file=ofh)
        print(f'`define {"GRO_CYCLE":16s} {spi_groc:5.1f} // SPI : GRO clock period',file=ofh)
        print(f'`define {"RO_GRID_NUM":16s} {spi_ro_num:5d} // SPI : number of ro',file=ofh)
        print(f'`define {"RO_EN_COUNT_BITS":16s} {spi_ro_enc:5d} // SPI : ro enable count bitwidth',file=ofh)
        print(f'Vsim definition header was saved into {out} ...')
        ofh.close()
        return True
        
    def envSetupSPICE(self):
        if os.path.exists(self.cfg['outPath']+'/spice')==False:
            print(f"ERROR! the specified output path {self.cfg['outPath']} does not exist")
            return False
        print('INFO: setup SPICE environment ...')
        outPath = self.cfg['outPath']
        # LPE merge
        if self.cfg.get('lpe',{})!={}:
            lpeL = " ".join(self.cfg.get('lpe',{}))
            os.system(f'cat {lpeL} > {outPath}/spice/merge.spx')  
            print(f'merge {lpeL} into {outPath}/spice/merge.spx')
            print(f'===>{lpeL}\n\n')
        return True

    def initMakefile(self):
        if os.path.exists(self.cfg.get('outPath',''))==False:
            print(f"ERROR! the specified output path {self.cfg['outPath']} does not exist")
            return False
        print('INFO: create makefile ...')
        out = f"{self.cfg['outPath']}/makefile"
        ofh = open(out,'w') 
        vlog = " \\\n            ".join([v for v in self.cfg.get('vlog',{})])
        print(f'# auto build by GRO compiler {self.date}',file=ofh)
        print(f"CONFIG     = {self.cfg.get('configFile','')}",file=ofh)
        print(f"OUTPATH    = {self.cfg.get('outPath','')}",file=ofh)
        print(f"TOP        = {self.cfg.get('top','')}",file=ofh)
        print(f"TARGETCOND = {self.cfg.get('targetCond','')}",file=ofh)
        print(f"VLOG       = {vlog}",file=ofh)
        print("",file=ofh)
        print("init: ${CONFIG}",file=ofh)
        print("\t@echo \"init makefile ...\"",file=ofh)
        print("",file=ofh)
        # linberty JSON
        print("json: ${CONFIG}",file=ofh)
        print("\t@echo \"build liberty JSON ...\"",file=ofh)
        print(f"\t{self.src}/gro.py \\",file=ofh)
        print("            -config ${CONFIG} \\",file=ofh)
        print("            -outPath ${OUTPATH} \\",file=ofh)
        print("            -target ${TARGETCOND} \\",file=ofh)
        print("            -initLib",file=ofh)
        print("",file=ofh)
        # RO design 
        print("gro: ${CONFIG}",file=ofh)
        print("\t@echo \"RO design generation\"",file=ofh)
        print(f"\t{self.src}/gro.py \\",file=ofh)
        print("            -config ${CONFIG} \\",file=ofh)
        print("            -outPath ${OUTPATH} \\",file=ofh)
        print("            -target ${TARGETCOND} \\",file=ofh)
        print("            -buildRO",file=ofh)
        print("",file=ofh)
        # DC synthesis
        print("syn:",file=ofh)
        print("\t@echo \"synthesis\"",file=ofh)
        print("\tcd ${OUTPATH}/synthesis; \\",file=ofh)
        print("        dc_shell -f main.tcl",file=ofh)
        print("",file=ofh)
        # testbench
        print("sim:",file=ofh)
        print("\t@echo \"gate-simulation\"",file=ofh)
        print("\tcd ${OUTPATH}/testbench; \\",file=ofh)
        print("        vcs ../verilog/define.vh tb.v ../synthesis/Netlist/${TOP}_syn.v \\",file=ofh)
        print("            ${VLOG} \\",file=ofh)
        print("            +define+SDF -fsdb -full64 +maxdelays",file=ofh)
        print("",file=ofh)
        # gate-sim waveform
        print("simv:",file=ofh)
        print("\t@echo \"gate-simulation\"",file=ofh)
        print("\tcd ${OUTPATH}/testbench; \\",file=ofh)
        print("        ./simv",file=ofh)
        print("",file=ofh)
        # RO SPICE circuit
        print("v2lvs:",file=ofh)
        print("\t@echo \"RO SPICE circuit\"",file=ofh)
        print("\tcd ${OUTPATH}/spice; \\",file=ofh)
        print("        v2lvs -addpin VDD -addpin VSS -addpin VBP -addpin VBN \\",file=ofh)
        print("            -v ../synthesis/Netlist/${TOP}_syn.v \\",file=ofh)
        print("            -lsr merge.spx \\",file=ofh)
        print("            -o ${TOP}.spi -i",file=ofh)
        print("",file=ofh)
        # SPICE simulation
        print("hspice:",file=ofh)
        print("\t@echo \"SPICE simulation\"",file=ofh)
        print("\tcd ${OUTPATH}/spice; \\",file=ofh)
        print("        hspice sim_GRO.sp -mt 20 -hpp -merge -o",file=ofh)
        print("",file=ofh)
        # SPICE result
        print("hspicev:",file=ofh)
        print("\t@echo \"SPICE result\"",file=ofh)
        print("\tcd ${OUTPATH}/spice; \\",file=ofh)
        print("        cat *.mt* |awk '/e-/{print \"DL\"$$3,$$1}'",file=ofh)
        print("",file=ofh)
        ofh.close()
        return True
        
    # load & convert CCS liberty to JSON 
    def initLibJSON(self,condL=None):
        outPath = self.cfg['outPath']+'/metric/JSON'
        if self.cfg.get('lib',{})=={}:
            print('ERROR: please specify iberty in the configuration first')
            return False
        if os.path.exists(outPath)==False:
            print('ERROR! please create project directory first')
            return False
        condL = list(self.cfg.get('lib',{}).keys()) if condL==None else condL
        print(f'init JSON DB: corner list {condL}')
        for cond in condL:
            for lib in self.cfg.get('lib').get(cond):
                lnode = self.loadLib(lib)
                json = lib.split('/')[-1].split('.')[0]+'.json'
                self.dump_json(lnode, out=f'{outPath}/{json}')
                if self.libDB.get(cond)==None:
                    self.libDB[cond] = [lnode]
                else:
                    self.libDB[cond]+= [lnode]
        # enumerate JSON DB
        for cond in self.libDB.keys():
            print(f'{cond:15s}: {len(self.libDB[cond])} libraries')
            for lnode in self.libDB[cond]:
                print(f'  {lnode["library"]}: {len(lnode["cell"])} cells')
        return True
    
    def commitConfig(self): # if user waive initLibJSON flow
        '''perform integrity checks for RO components and setup successive flow environments'''
        if self.cfg.get('delayLine')==None or self.cfg.get('lib')==None:
            print('ERROR: please specify config file first')
            return False
        
        if self.cfg.get('targetCond','TT') not in self.cfg.get('lib',{}).keys():
            print(f"ERROR: target condition should be one of {list(self.cfg.get('lib',{}).keys())}")
            return False

        # load JSON DB, need to perform initLibJSON first
        if self.libDB=={}:
            target = self.cfg.get('targetCond','TT')
            self.libDB[target] = []
            print(f'INFO! auto commit liberty from JSON DB, target cond: {target} ...')
            libL = list(self.cfg.get('lib',{}).get(target))
            for lib in libL:
                json = os.path.basename(lib).split('.')[0]+'.json'
                lnode = self.loadLib(self.cfg['outPath']+'/metric/JSON/'+json)
                self.libDB[target] += [lnode]
        
        # RO component integrity check
        if self.integrityCheck(self.cfg.get('targetCond','TT'))==False: # GRO component integrity check, cfg update
            print('ERROR: please check the liberty integrity, containing all necessary GRO components')
            return False
        
        # build successive sub flows
        try:
            ret = all([
                self.envSetupDC(),
                self.envSetupVsim(),
                self.envSetupSPICE()
                ])
        except:
            print('ERROR: some environments (DC, Vsim, SPICE) were not successfully established')
            ret = False
        return ret
    
    def initProjectDirectory(self):
        print(f'create GRO project directory {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        outPath = self.cfg['outPath']
        #src = os.path.dirname(os.path.realpath(self.src)) # path of gro.py
        env = os.path.realpath(f'{self.src}/../env') # path of GRO/src project
        print(f'Program Root = {self.src}')
        print(f'Environment Dir = {env}')
        print(f'Output Dir = {outPath}')
        try:
            if os.path.exists(f'{outPath}')==False:
                os.mkdir(f'{outPath}')
            if os.path.exists(f'{outPath}/metric')==False:
                os.mkdir(f'{outPath}/metric')
                os.mkdir(f'{outPath}/metric/JSON')
            if os.path.exists(f'{outPath}/verilog')==False:
                os.mkdir(f'{outPath}/verilog')
            if os.path.exists(f'{outPath}/synthesis')==False:
                os.mkdir(f'{outPath}/synthesis')
            if os.path.exists(f'{outPath}/testbench')==False:
                os.mkdir(f'{outPath}/testbench')
            if os.path.exists(f'{outPath}/spice')==False:
                os.mkdir(f'{outPath}/spice')
            os.system(f'cp {env}/verilog/* {outPath}/verilog -r')
            os.system(f'cp {env}/synthesis/* {outPath}/synthesis -r')  
            os.system(f'cp {env}/testbench/* {outPath}/testbench -r')
        except Exception as e:
            print(e)
            return False
        return True
    
    # user command line argument parser
    def parseArguments(self,args):
        script,*argv = args
        ii,argc = 0,len(argv)
        pdata,code = {},True
        while ii<argc:
            v = argv[ii]
            if v[:2]=='-h':
                print('Grid RO Compiler')
                print('usage:\n    python3 {script} -cfg <config.cfg> [options]\n')
                print('options:')
                print('-configFile   : configuration file')
                print('-outPath      : output directory')
                print('[-targetCond] : target liberty condition, "default" will be applied without configuration')
                print('[-initProj]   : create & initiate RO project directory only')
                print('[-initLib]    : convert liberty CCS (defined in the config) to JSON DB')
                print('[-buildRO]    : generate RO design and the successive synthesis, gate-sim and spice flows')
                print('[-help]       : show command usage\n')
                code = False
                break
            elif v[:2]=='-o':
                ii += 1
                pdata['outPath'] = os.path.realpath(argv[ii])
            elif v[:2]=='-t':
                ii += 1
                pdata['targetCond'] = argv[ii]
            elif v[:2]=='-c':
                ii += 1
                pdata['configFile'] = argv[ii]
            elif v[:6]=='-initP':
                pdata['initProj'] = True
                print('INFO: initiate RO project directory')
            elif v[:6]=='-initL':
                pdata['initLib'] = True
                print('INFO: initiate liberty JSON')
            elif v[:2]=='-b':
                print('INFO: build RO design, and the successive flow environments')
                pdata['buildRO'] = True
            else: # remaining argument as the configuration
                pdata['configFile'] = argv[ii]
            ii += 1
        # consistency check
        if pdata.get('configFile')==None or os.path.exists(pdata['configFile'])==False:
            print('ERROR! please spefify a configuration first')
            code = False
        if pdata.get('outPath')==None: # auto patch with current directory
            #pdata['outPath'] = "." # os.path.dirname(__file__) # where the script is launch
            pdata['outPath'] = os.path.realpath('./') + '/RO_tmp_output'
            print('ERROR! please spefify a output path for RO project')
            code = False
        # patch config from command arguments
        if code==True:
            pdata['configFile'] = os.path.realpath(pdata['configFile'])           
            self.cfg['targetCond'] = pdata.get('targetCond','default') 
            self.cfg['outPath'] = self.cfg.get('outPath',pdata['outPath'])
            self.cfg['configFile'] = self.cfg.get('configFile',pdata['configFile'])
            self.cfg['SPI'] = dict({}) # overwrite from configFile
            self.cfg['vlog'] = dict({}) # overwrite from configFile
            self.cfg['db'] = dict({}) # overwrite from configFile
            self.cfg['lpe'] = dict({}) # overwrite from configFile
            self.cfg['model_card'] = dict({}) # overwrite from configFile
        #print('\nuser parameters:')  
        #print("\n".join([f'{v:10s}: {pdata[v]}' for v in pdata]))
        return code,pdata


#%% verify
if False: #__name__ == '__main__':

    gro = ROCompiler()
    
    # concert liberty to JSON
    lnode = gro.read_lib('liberty_ccs.lib')
    gro.dump_json(lnode,'liberty_ccs.json')
    
    # query GRO components with cell_footprint
    lnode = gro.loadLib('liberty_ccs.json')
    nd2 = gro.queryCellWithFP(lnode,fp='nd2d1')
    nr2 = gro.queryCellWithFP(lnode,fp='nr2d1')
    dff = gro.queryCellWithFP(lnode,fp='dfqd1')
    inv = gro.queryCellWithFP(lnode,fp='invd1')
    
    # query GRO components with regular expression
    nd2 = gro.queryCellWithRE(lnode,exp='ND2D1')
    nr2 = gro.queryCellWithRE(lnode,exp='NR2D1')
    dff = gro.queryCellWithRE(lnode,exp='DFQD1')
    inv = gro.queryCellWithRE(lnode,exp='INVD1')
    
    #%% grab base gate 
    gro.baseGateInfo(lnode,cell='ND2D1')
    cfg = gro.loadConfig('config.f')
    cfg = gro.integrityCheck(lnode) # GRO component integrity based on the configuration
    
    #%% GRO compiler 
    cfg = gro.loadConfig('config.f')
    gro.initProjectDirectory()
    gro.initMakefile()
    gro.initLibJSON() # build library JSON DB
    gro.commitConfig() # start from liberty JSON without initLibJSON
    gro.compileGRO()
    