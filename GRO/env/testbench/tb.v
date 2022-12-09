`timescale 1ns/1ps
`define SDFFILE "../synthesis/Netlist/GRO_TOP_syn.sdf"
module tb ();
    reg	                 EN;
    reg                  RSTN;
    reg  [`SEL_BITS-1:0] SEL;
    reg                  CLK;
    wire [`RPC_BITS-1:0] COUNT;
    wire                 RO;
    integer              i;
    real                 period;
    initial begin
        $fsdbDumpfile("tb.fsdb");
        $fsdbDumpvars(0, tb, "+mda");
    end

    initial begin
        `ifdef SDF
            $sdf_annotate(`SDFFILE, u_ro_top);
        `endif
    end

    GRO_TOP u_ro_top (
        .COUNT              (COUNT             ),
        .RSTN               (RSTN              ),
        .SEL                (SEL               ),
        .EN                 (EN                ),
        .RO                 (RO                )
    );
    always #(`GRO_CYCLE/2) CLK = ~CLK;
    initial begin
        CLK            = 0;
        EN             = 0;
        SEL	           = 0;
        for (i = 0; i < `DL_NUM; i = i + 1) begin
            reset_task;
            test_ring;
        end
        $finish;
    end
    
    task reset_task; begin
        @(negedge CLK) RSTN = 0;
        @(negedge CLK) RSTN = 1;
    end endtask
    
    task test_ring; begin
        EN = 1;
        #(4096*`GRO_CYCLE);
        EN = 0;
        period = (4096.0/COUNT)*`GRO_CYCLE*1000;
        $display("DL%d measured period : %f ps", SEL, period);
        SEL = SEL + 1;
    end endtask
endmodule
