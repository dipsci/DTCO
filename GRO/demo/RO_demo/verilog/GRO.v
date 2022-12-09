// auto build by GRO compiler 09/12/2022 14:04:48
module SDLI(EN, SEL, ENO);
output reg [7:0] ENO;
input [2:0] SEL;
input EN;
always @(*) begin
    if (EN) begin
        case(SEL)
            0: ENO = 8'b00000001;
            1: ENO = 8'b00000010;
            2: ENO = 8'b00000100;
            3: ENO = 8'b00001000;
            4: ENO = 8'b00010000;
            5: ENO = 8'b00100000;
            6: ENO = 8'b01000000;
            7: ENO = 8'b10000000;
            default: ENO = 8'b00000000;
        endcase
    end else begin
        ENO = 8'b00000000;
    end
end
endmodule

module SDLO (EN, SEL, ENO);
output reg ENO;
input [7:0] EN;
input [2:0] SEL;
always @ (*) begin
case (SEL)
    3'b000 : ENO = EN[0];
    3'b001 : ENO = EN[1];
    3'b010 : ENO = EN[2];
    3'b011 : ENO = EN[3];
    3'b100 : ENO = EN[4];
    3'b101 : ENO = EN[5];
    3'b110 : ENO = EN[6];
    3'b111 : ENO = EN[7];
    default : ENO = EN[0];
    endcase
end
endmodule

module RippleCounter(CLK, COUNT, RSTN);
input CLK, RSTN;
output [15:0] COUNT;
DFCNQD1LVT u_dff0 (
    .Q(COUNT[0]),
    .CP(CLK), .D(n_inv0_o), .CDN(RSTN));
INVD1LVT u_inv0 (.ZN(n_inv0_o), .I(COUNT[0]));
DFCNQD1LVT u_dff1 (
    .Q(COUNT[1]),
    .CP(n_inv0_o), .D(n_inv1_o), .CDN(RSTN));
INVD1LVT u_inv1 (.ZN(n_inv1_o), .I(COUNT[1]));
DFCNQD1LVT u_dff2 (
    .Q(COUNT[2]),
    .CP(n_inv1_o), .D(n_inv2_o), .CDN(RSTN));
INVD1LVT u_inv2 (.ZN(n_inv2_o), .I(COUNT[2]));
DFCNQD1LVT u_dff3 (
    .Q(COUNT[3]),
    .CP(n_inv2_o), .D(n_inv3_o), .CDN(RSTN));
INVD1LVT u_inv3 (.ZN(n_inv3_o), .I(COUNT[3]));
DFCNQD1LVT u_dff4 (
    .Q(COUNT[4]),
    .CP(n_inv3_o), .D(n_inv4_o), .CDN(RSTN));
INVD1LVT u_inv4 (.ZN(n_inv4_o), .I(COUNT[4]));
DFCNQD1LVT u_dff5 (
    .Q(COUNT[5]),
    .CP(n_inv4_o), .D(n_inv5_o), .CDN(RSTN));
INVD1LVT u_inv5 (.ZN(n_inv5_o), .I(COUNT[5]));
DFCNQD1LVT u_dff6 (
    .Q(COUNT[6]),
    .CP(n_inv5_o), .D(n_inv6_o), .CDN(RSTN));
INVD1LVT u_inv6 (.ZN(n_inv6_o), .I(COUNT[6]));
DFCNQD1LVT u_dff7 (
    .Q(COUNT[7]),
    .CP(n_inv6_o), .D(n_inv7_o), .CDN(RSTN));
INVD1LVT u_inv7 (.ZN(n_inv7_o), .I(COUNT[7]));
DFCNQD1LVT u_dff8 (
    .Q(COUNT[8]),
    .CP(n_inv7_o), .D(n_inv8_o), .CDN(RSTN));
INVD1LVT u_inv8 (.ZN(n_inv8_o), .I(COUNT[8]));
DFCNQD1LVT u_dff9 (
    .Q(COUNT[9]),
    .CP(n_inv8_o), .D(n_inv9_o), .CDN(RSTN));
INVD1LVT u_inv9 (.ZN(n_inv9_o), .I(COUNT[9]));
DFCNQD1LVT u_dff10 (
    .Q(COUNT[10]),
    .CP(n_inv9_o), .D(n_inv10_o), .CDN(RSTN));
INVD1LVT u_inv10 (.ZN(n_inv10_o), .I(COUNT[10]));
DFCNQD1LVT u_dff11 (
    .Q(COUNT[11]),
    .CP(n_inv10_o), .D(n_inv11_o), .CDN(RSTN));
INVD1LVT u_inv11 (.ZN(n_inv11_o), .I(COUNT[11]));
DFCNQD1LVT u_dff12 (
    .Q(COUNT[12]),
    .CP(n_inv11_o), .D(n_inv12_o), .CDN(RSTN));
INVD1LVT u_inv12 (.ZN(n_inv12_o), .I(COUNT[12]));
DFCNQD1LVT u_dff13 (
    .Q(COUNT[13]),
    .CP(n_inv12_o), .D(n_inv13_o), .CDN(RSTN));
INVD1LVT u_inv13 (.ZN(n_inv13_o), .I(COUNT[13]));
DFCNQD1LVT u_dff14 (
    .Q(COUNT[14]),
    .CP(n_inv13_o), .D(n_inv14_o), .CDN(RSTN));
INVD1LVT u_inv14 (.ZN(n_inv14_o), .I(COUNT[14]));
DFCNQD1LVT u_dff15 (
    .Q(COUNT[15]),
    .CP(n_inv14_o), .D(n_inv15_o), .CDN(RSTN));
INVD1LVT u_inv15 (.ZN(n_inv15_o), .I(COUNT[15]));
endmodule

// DL0,2000ps, ND2D1LVT: I:['A1', 'A2'], O:['ZN'], delay=51.00ps, stage:18
module DL0_ND2D1LVT ( .EN(EN), .RO(RO) );
input EN;
output RO;
wire [20:0] NET;
ND2D1LVT u_enable ( // 51.0ps
    .A2(EN),
    .A1(NET[19]),
    .ZN(NET[1]) );
INVD1LVT u_inv (
    .I(NET[19]),
    .ZN(RO) );
ND2D1LVT u_u1 ( .A1(NET[1]), .A2(NET[1]), .ZN(NET[2]) ); // 102.00ps
ND2D1LVT u_u2 ( .A1(NET[2]), .A2(NET[2]), .ZN(NET[3]) ); // 153.00ps
ND2D1LVT u_u3 ( .A1(NET[3]), .A2(NET[3]), .ZN(NET[4]) ); // 204.00ps
ND2D1LVT u_u4 ( .A1(NET[4]), .A2(NET[4]), .ZN(NET[5]) ); // 255.00ps
ND2D1LVT u_u5 ( .A1(NET[5]), .A2(NET[5]), .ZN(NET[6]) ); // 306.00ps
ND2D1LVT u_u6 ( .A1(NET[6]), .A2(NET[6]), .ZN(NET[7]) ); // 357.00ps
ND2D1LVT u_u7 ( .A1(NET[7]), .A2(NET[7]), .ZN(NET[8]) ); // 408.00ps
ND2D1LVT u_u8 ( .A1(NET[8]), .A2(NET[8]), .ZN(NET[9]) ); // 459.00ps
ND2D1LVT u_u9 ( .A1(NET[9]), .A2(NET[9]), .ZN(NET[10]) ); // 510.00ps
ND2D1LVT u_u10 ( .A1(NET[10]), .A2(NET[10]), .ZN(NET[11]) ); // 561.00ps
ND2D1LVT u_u11 ( .A1(NET[11]), .A2(NET[11]), .ZN(NET[12]) ); // 612.00ps
ND2D1LVT u_u12 ( .A1(NET[12]), .A2(NET[12]), .ZN(NET[13]) ); // 663.00ps
ND2D1LVT u_u13 ( .A1(NET[13]), .A2(NET[13]), .ZN(NET[14]) ); // 714.00ps
ND2D1LVT u_u14 ( .A1(NET[14]), .A2(NET[14]), .ZN(NET[15]) ); // 765.00ps
ND2D1LVT u_u15 ( .A1(NET[15]), .A2(NET[15]), .ZN(NET[16]) ); // 816.00ps
ND2D1LVT u_u16 ( .A1(NET[16]), .A2(NET[16]), .ZN(NET[17]) ); // 867.00ps
ND2D1LVT u_u17 ( .A1(NET[17]), .A2(NET[17]), .ZN(NET[18]) ); // 918.00ps
ND2D1LVT u_u18 ( .A1(NET[18]), .A2(NET[18]), .ZN(NET[19]) ); // 969.00ps
endmodule

// DL1,2000ps, NR2D1LVT: I:['A1', 'A2'], O:['ZN'], delay=78.00ps, stage:12
module DL1_NR2D1LVT ( .EN(EN), .RO(RO) );
input EN;
output RO;
wire [14:0] NET;
ND2D1LVT u_enable ( // 51.0ps
    .A2(EN),
    .A1(NET[13]),
    .ZN(NET[1]) );
INVD1LVT u_inv (
    .I(NET[13]),
    .ZN(RO) );
NR2D1LVT u_u1 ( .A1(NET[1]), .A2(NET[1]), .ZN(NET[2]) ); // 129.00ps
NR2D1LVT u_u2 ( .A1(NET[2]), .A2(NET[2]), .ZN(NET[3]) ); // 207.00ps
NR2D1LVT u_u3 ( .A1(NET[3]), .A2(NET[3]), .ZN(NET[4]) ); // 285.00ps
NR2D1LVT u_u4 ( .A1(NET[4]), .A2(NET[4]), .ZN(NET[5]) ); // 363.00ps
NR2D1LVT u_u5 ( .A1(NET[5]), .A2(NET[5]), .ZN(NET[6]) ); // 441.00ps
NR2D1LVT u_u6 ( .A1(NET[6]), .A2(NET[6]), .ZN(NET[7]) ); // 519.00ps
NR2D1LVT u_u7 ( .A1(NET[7]), .A2(NET[7]), .ZN(NET[8]) ); // 597.00ps
NR2D1LVT u_u8 ( .A1(NET[8]), .A2(NET[8]), .ZN(NET[9]) ); // 675.00ps
NR2D1LVT u_u9 ( .A1(NET[9]), .A2(NET[9]), .ZN(NET[10]) ); // 753.00ps
NR2D1LVT u_u10 ( .A1(NET[10]), .A2(NET[10]), .ZN(NET[11]) ); // 831.00ps
NR2D1LVT u_u11 ( .A1(NET[11]), .A2(NET[11]), .ZN(NET[12]) ); // 909.00ps
NR2D1LVT u_u12 ( .A1(NET[12]), .A2(NET[12]), .ZN(NET[13]) ); // 987.00ps
endmodule

// DL2,2000ps, AN2D1LVT: I:['A1', 'A2'], O:['Z'], delay=89.00ps, stage:10
module DL2_AN2D1LVT ( .EN(EN), .RO(RO) );
input EN;
output RO;
wire [12:0] NET;
ND2D1LVT u_enable ( // 51.0ps
    .A2(EN),
    .A1(NET[11]),
    .ZN(NET[1]) );
INVD1LVT u_inv (
    .I(NET[11]),
    .ZN(RO) );
AN2D1LVT u_u1 ( .A1(NET[1]), .A2(NET[1]), .Z(NET[2]) ); // 140.00ps
AN2D1LVT u_u2 ( .A1(NET[2]), .A2(NET[2]), .Z(NET[3]) ); // 229.00ps
AN2D1LVT u_u3 ( .A1(NET[3]), .A2(NET[3]), .Z(NET[4]) ); // 318.00ps
AN2D1LVT u_u4 ( .A1(NET[4]), .A2(NET[4]), .Z(NET[5]) ); // 407.00ps
AN2D1LVT u_u5 ( .A1(NET[5]), .A2(NET[5]), .Z(NET[6]) ); // 496.00ps
AN2D1LVT u_u6 ( .A1(NET[6]), .A2(NET[6]), .Z(NET[7]) ); // 585.00ps
AN2D1LVT u_u7 ( .A1(NET[7]), .A2(NET[7]), .Z(NET[8]) ); // 674.00ps
AN2D1LVT u_u8 ( .A1(NET[8]), .A2(NET[8]), .Z(NET[9]) ); // 763.00ps
AN2D1LVT u_u9 ( .A1(NET[9]), .A2(NET[9]), .Z(NET[10]) ); // 852.00ps
AN2D1LVT u_u10 ( .A1(NET[10]), .A2(NET[10]), .Z(NET[11]) ); // 941.00ps
endmodule

// DL3,2000ps, OR2D1LVT: I:['A1', 'A2'], O:['Z'], delay=90.00ps, stage:10
module DL3_OR2D1LVT ( .EN(EN), .RO(RO) );
input EN;
output RO;
wire [12:0] NET;
ND2D1LVT u_enable ( // 51.0ps
    .A2(EN),
    .A1(NET[11]),
    .ZN(NET[1]) );
INVD1LVT u_inv (
    .I(NET[11]),
    .ZN(RO) );
OR2D1LVT u_u1 ( .A1(NET[1]), .A2(NET[1]), .Z(NET[2]) ); // 141.00ps
OR2D1LVT u_u2 ( .A1(NET[2]), .A2(NET[2]), .Z(NET[3]) ); // 231.00ps
OR2D1LVT u_u3 ( .A1(NET[3]), .A2(NET[3]), .Z(NET[4]) ); // 321.00ps
OR2D1LVT u_u4 ( .A1(NET[4]), .A2(NET[4]), .Z(NET[5]) ); // 411.00ps
OR2D1LVT u_u5 ( .A1(NET[5]), .A2(NET[5]), .Z(NET[6]) ); // 501.00ps
OR2D1LVT u_u6 ( .A1(NET[6]), .A2(NET[6]), .Z(NET[7]) ); // 591.00ps
OR2D1LVT u_u7 ( .A1(NET[7]), .A2(NET[7]), .Z(NET[8]) ); // 681.00ps
OR2D1LVT u_u8 ( .A1(NET[8]), .A2(NET[8]), .Z(NET[9]) ); // 771.00ps
OR2D1LVT u_u9 ( .A1(NET[9]), .A2(NET[9]), .Z(NET[10]) ); // 861.00ps
OR2D1LVT u_u10 ( .A1(NET[10]), .A2(NET[10]), .Z(NET[11]) ); // 951.00ps
endmodule

// DL4,2000ps, INVD1LVT: I:['I'], O:['ZN'], delay=27.00ps, stage:36
module DL4_INVD1LVT ( .EN(EN), .RO(RO) );
input EN;
output RO;
wire [38:0] NET;
ND2D1LVT u_enable ( // 51.0ps
    .A2(EN),
    .A1(NET[37]),
    .ZN(NET[1]) );
INVD1LVT u_inv (
    .I(NET[37]),
    .ZN(RO) );
INVD1LVT u_u1 ( .I(NET[1]), .ZN(NET[2]) ); // 78.00ps
INVD1LVT u_u2 ( .I(NET[2]), .ZN(NET[3]) ); // 105.00ps
INVD1LVT u_u3 ( .I(NET[3]), .ZN(NET[4]) ); // 132.00ps
INVD1LVT u_u4 ( .I(NET[4]), .ZN(NET[5]) ); // 159.00ps
INVD1LVT u_u5 ( .I(NET[5]), .ZN(NET[6]) ); // 186.00ps
INVD1LVT u_u6 ( .I(NET[6]), .ZN(NET[7]) ); // 213.00ps
INVD1LVT u_u7 ( .I(NET[7]), .ZN(NET[8]) ); // 240.00ps
INVD1LVT u_u8 ( .I(NET[8]), .ZN(NET[9]) ); // 267.00ps
INVD1LVT u_u9 ( .I(NET[9]), .ZN(NET[10]) ); // 294.00ps
INVD1LVT u_u10 ( .I(NET[10]), .ZN(NET[11]) ); // 321.00ps
INVD1LVT u_u11 ( .I(NET[11]), .ZN(NET[12]) ); // 348.00ps
INVD1LVT u_u12 ( .I(NET[12]), .ZN(NET[13]) ); // 375.00ps
INVD1LVT u_u13 ( .I(NET[13]), .ZN(NET[14]) ); // 402.00ps
INVD1LVT u_u14 ( .I(NET[14]), .ZN(NET[15]) ); // 429.00ps
INVD1LVT u_u15 ( .I(NET[15]), .ZN(NET[16]) ); // 456.00ps
INVD1LVT u_u16 ( .I(NET[16]), .ZN(NET[17]) ); // 483.00ps
INVD1LVT u_u17 ( .I(NET[17]), .ZN(NET[18]) ); // 510.00ps
INVD1LVT u_u18 ( .I(NET[18]), .ZN(NET[19]) ); // 537.00ps
INVD1LVT u_u19 ( .I(NET[19]), .ZN(NET[20]) ); // 564.00ps
INVD1LVT u_u20 ( .I(NET[20]), .ZN(NET[21]) ); // 591.00ps
INVD1LVT u_u21 ( .I(NET[21]), .ZN(NET[22]) ); // 618.00ps
INVD1LVT u_u22 ( .I(NET[22]), .ZN(NET[23]) ); // 645.00ps
INVD1LVT u_u23 ( .I(NET[23]), .ZN(NET[24]) ); // 672.00ps
INVD1LVT u_u24 ( .I(NET[24]), .ZN(NET[25]) ); // 699.00ps
INVD1LVT u_u25 ( .I(NET[25]), .ZN(NET[26]) ); // 726.00ps
INVD1LVT u_u26 ( .I(NET[26]), .ZN(NET[27]) ); // 753.00ps
INVD1LVT u_u27 ( .I(NET[27]), .ZN(NET[28]) ); // 780.00ps
INVD1LVT u_u28 ( .I(NET[28]), .ZN(NET[29]) ); // 807.00ps
INVD1LVT u_u29 ( .I(NET[29]), .ZN(NET[30]) ); // 834.00ps
INVD1LVT u_u30 ( .I(NET[30]), .ZN(NET[31]) ); // 861.00ps
INVD1LVT u_u31 ( .I(NET[31]), .ZN(NET[32]) ); // 888.00ps
INVD1LVT u_u32 ( .I(NET[32]), .ZN(NET[33]) ); // 915.00ps
INVD1LVT u_u33 ( .I(NET[33]), .ZN(NET[34]) ); // 942.00ps
INVD1LVT u_u34 ( .I(NET[34]), .ZN(NET[35]) ); // 969.00ps
INVD1LVT u_u35 ( .I(NET[35]), .ZN(NET[36]) ); // 996.00ps
INVD1LVT u_u36 ( .I(NET[36]), .ZN(NET[37]) ); // 1023.00ps
endmodule

// DL5,2000ps, BUFFD1LVT: I:['I'], O:['Z'], delay=67.00ps, stage:14
module DL5_BUFFD1LVT ( .EN(EN), .RO(RO) );
input EN;
output RO;
wire [16:0] NET;
ND2D1LVT u_enable ( // 51.0ps
    .A2(EN),
    .A1(NET[15]),
    .ZN(NET[1]) );
INVD1LVT u_inv (
    .I(NET[15]),
    .ZN(RO) );
BUFFD1LVT u_u1 ( .I(NET[1]), .Z(NET[2]) ); // 118.00ps
BUFFD1LVT u_u2 ( .I(NET[2]), .Z(NET[3]) ); // 185.00ps
BUFFD1LVT u_u3 ( .I(NET[3]), .Z(NET[4]) ); // 252.00ps
BUFFD1LVT u_u4 ( .I(NET[4]), .Z(NET[5]) ); // 319.00ps
BUFFD1LVT u_u5 ( .I(NET[5]), .Z(NET[6]) ); // 386.00ps
BUFFD1LVT u_u6 ( .I(NET[6]), .Z(NET[7]) ); // 453.00ps
BUFFD1LVT u_u7 ( .I(NET[7]), .Z(NET[8]) ); // 520.00ps
BUFFD1LVT u_u8 ( .I(NET[8]), .Z(NET[9]) ); // 587.00ps
BUFFD1LVT u_u9 ( .I(NET[9]), .Z(NET[10]) ); // 654.00ps
BUFFD1LVT u_u10 ( .I(NET[10]), .Z(NET[11]) ); // 721.00ps
BUFFD1LVT u_u11 ( .I(NET[11]), .Z(NET[12]) ); // 788.00ps
BUFFD1LVT u_u12 ( .I(NET[12]), .Z(NET[13]) ); // 855.00ps
BUFFD1LVT u_u13 ( .I(NET[13]), .Z(NET[14]) ); // 922.00ps
BUFFD1LVT u_u14 ( .I(NET[14]), .Z(NET[15]) ); // 989.00ps
endmodule

module GRO_TOP (EN, RSTN, SEL, COUNT, RO);
input EN;
input RSTN;
input [2:0] SEL;
output [15:0] COUNT;
output RO;
wire [7:0] n_sdli_o;
wire [7:0] n_sdlo_i;
SDLI u_sdli (.EN(EN), .SEL(SEL), .ENO(n_sdli_o));
SDLO u_sdlo (.EN(n_sdlo_i), .SEL(SEL), .ENO(RO));
RippleCounter u_count (.CLK(RO), .COUNT(COUNT), .RSTN(RSTN));
DL0_ND2D1LVT u_DL0 (.RO(n_sdlo_i[0]), .EN(n_sdli_o[0]));
DL1_NR2D1LVT u_DL1 (.RO(n_sdlo_i[1]), .EN(n_sdli_o[1]));
DL2_AN2D1LVT u_DL2 (.RO(n_sdlo_i[2]), .EN(n_sdli_o[2]));
DL3_OR2D1LVT u_DL3 (.RO(n_sdlo_i[3]), .EN(n_sdli_o[3]));
DL4_INVD1LVT u_DL4 (.RO(n_sdlo_i[4]), .EN(n_sdli_o[4]));
DL5_BUFFD1LVT u_DL5 (.RO(n_sdlo_i[5]), .EN(n_sdli_o[5]));
assign n_sdlo_i[6] = 0;
assign n_sdlo_i[7] = 0;
endmodule

