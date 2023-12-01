#!/bin/python3
'''
Command line argument parser for Grid RO (GRO) Compiler v0.1 build 2022/6

Copyright (C) 2022, by DigWise Technology Corpopation, LTD.
This software tool is independently developed and owned by DigWise Technology Co., Ltd.
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

#%% 
import sys
#sys.path.append('C:/Home/Projects/Pypi/DTCO')
#pip install DTCO
from GRO import ROCompiler 

if __name__ == '__main__':
    argv = sys.argv
else: # test mode
    argv = ['this',
            '-config','config_demo.f',
            '-outDir','GRO',
            '-target','TT',
            '-initProj','-initLib','-buildRO']
    
# init GRO instance
gro = ROCompiler()

#%% parse command line
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

