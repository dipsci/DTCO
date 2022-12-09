
create_clock $CLOCK -period $CLOCK_PERIOD
set_ideal_network [get_ports $CLOCK]
set_dont_touch_network [get_ports $CLOCK]
set_dont_touch [get_cells u_DL*]
set_clock_uncertainty  0.05    $CLOCK
set_clock_transition   0.075   $CLOCK
set_input_delay  [ expr $CLOCK_PERIOD*0.25 ] -clock $CLOCK [all_inputs]
set_output_delay [ expr $CLOCK_PERIOD*0.25 ] -clock $CLOCK [all_outputs]
set_driving_cell -lib_cell $DRIVING_CELL -pin $DRIVING_PIN -no_design_rule [all_inputs]
set_load $LOAD [all_outputs]

set_boundary_optimization $DESIGN false
check_design
uniquify
set_fix_multiple_port_nets -all -buffer_constants  [get_designs *]
set_fix_hold [all_clocks]
