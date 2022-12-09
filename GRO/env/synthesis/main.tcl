source ../verilog/config.tcl 
puts "==============================================================="
parray groCfg::DC
puts "==============================================================="

# Setting environmen #
sh rm -rf Nelist
sh rm -rf Report
sh mkdir -p Netlist
sh mkdir -p Report
set hdlin_translate_off_skip_text "TRUE"
set edifout_netlist_only "TRUE"
set verilogout_no_tri true
set plot_command {lpr -Plw}
set hdlin_auto_save_templates "TRUE"
set compile_fix_multiple_port_nets "TRUE"
## Set below to ignore Techonology File Error TFCHK-009 in DCT
set ignore_tf_error true

##  Set libraries to use for synthesis.
set libName   $groCfg::DC(libName)
set topModule $groCfg::DC(topModule) 
set sdc "design.sdc"
#set search_path [subst { . }]
#set link_library  [subst {./${libName}.db}]
#set target_library [subst {./${libName}.db}]
#set symbol_library [subst {./${libName}.sdb}]
#set search_path $groCfg::DC(libPath)
#set link_library [groCfg::joinExt $groCfg::DC(libName) .db]
#set target_library [groCfg::joinExt $groCfg::DC(libName) .db]
#set symbol_library [groCfg::joinExt $groCfg::DC(libName) .sdb]
set search_path $groCfg::DC(libPath)
set link_library $groCfg::DC(libName) 
set target_library $groCfg::DC(libName) 
set symbol_library $groCfg::DC(libName)
set DESIGN ${topModule}
analyze -format sverilog ./flist.v
elaborate $DESIGN
link

##  Read multiple RTL files
set SPI_CLOCK  "SCK"
set SPI_CLOCK_PERIOD 3333.3 
set INT_CLOCK  "CK_i"
set INT_CLOCK_PERIOD 5.0
set RO_CLOCK [get_pins */u_gro/u_sdlo/ENO]
set RO_CLOCK_PERIOD $groCfg::DC(clockPeriod) 
set DRIVING_CELL $groCfg::DC(drivingCell)
set DRIVING_PIN $groCfg::DC(drivingPin)
set LOAD $groCfg::DC(cload)

puts "==============================================================="
puts $libName
puts $topModule
puts $SPI_CLOCK_PERIOD
puts $INT_CLOCK_PERIOD
puts $RO_CLOCK_PERIOD
puts "==============================================================="

##  Constrain timing for the design.
source -echo -verbose ./$sdc 
set_operating_conditions -max $groCfg::DC(opConditions)

##  Compile
compile
compile -inc

## Report
change_names -rules verilog -hierarchy 
report_area > Report/${topModule}_area.log
report_timing >> Report/${topModule}_timing.log
report_power >> Report/${topModule}_power.log
report_qor >> Report/${topModule}_qor.log
report_lib ${libName} >>  Report/${topModule}_lib.log

## Change Naming Rule
set bus_inference_style "%s\[%d\]"
set bus_naming_style "%s\[%d\]"
set hdlout_internal_busses true
change_names -hierarchy -rule verilog
define_name_rules name_rule -allowed "a-z A-Z 0-9 _" -max_length 255 -type cell
define_name_rules name_rule -allowed "a-z A-Z 0-9 _[]" -max_length 255 -type net
define_name_rules name_rule -map {{"\\*cell\\*" "cell"}}
define_name_rules name_rule -case_insensitive
change_name -hierarchy -rules name_rule

##  Write gate level verilog file for the design.
write_sdc  Netlist/${topModule}_syn.sdc -version 1.9
write -hierarchy -format verilog -output Netlist/${topModule}_syn.v
write_sdf -version 2.1 Netlist/${topModule}_syn.sdf
quit
