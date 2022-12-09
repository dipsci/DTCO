# configuration example

lib = { # CCS liberty or JSON
    'TT': ['C:/Home/Projects/00EDA/GRO/latest/demo/JSON/libertydemo.json'],
}

enGate = { # ND2 enable gate, as base gate as well
    'name' :'ND2D1LVT',
}
      
invGate = { # inversion for ripple counter
    'name': 'INVD1LVT',
}

cntDFF = { # counter DFF
    'name'  : 'DFCNQD1LVT', 
    'bits'  : 16,
    'clear' : 'CDN', # overwrite with user specified pin_name 
}

delayLine = { # num: cell, period(ps)  
    'DL0' : ['ND2D1LVT', 2000],
    'DL1' : ['NR2D1LVT', 2000],
    'DL2' : ['AN2D1LVT', 2000],
    'DL3' : ['OR2D1LVT', 2000],
    'DL4' : ['INVD1LVT', 2000],
    'DL5' : ['BUFFD1LVT', 2000],
}

## successive execution flows
#db = { # DC synthesis setup
#}
#vlog = { # testbench for gate simulation
#}

## SPICE circuit extraction & simulation
#lpe = { # LPE SPICE netlist
#}
#model_card = { # SPICE Model
#    'TT':''
#}

