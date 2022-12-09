# auto build by GRO compiler 09/12/2022 14:04:48
CONFIG     = C:\Home\Projects\Pypi\DTCO\GRO\demo\config_demo.f
OUTPATH    = C:\Home\Projects\Pypi\DTCO\GRO\demo\RO_demo
TOP        = GRO_TOP
TARGETCOND = TT
VLOG       = 

init: ${CONFIG}
	@echo "init makefile ..."

json: ${CONFIG}
	@echo "build liberty JSON ..."
	C:\Home\Projects\Pypi\DTCO\GRO/gro.py \
            -config ${CONFIG} \
            -outPath ${OUTPATH} \
            -target ${TARGETCOND} \
            -initLib

gro: ${CONFIG}
	@echo "RO design generation"
	C:\Home\Projects\Pypi\DTCO\GRO/gro.py \
            -config ${CONFIG} \
            -outPath ${OUTPATH} \
            -target ${TARGETCOND} \
            -buildRO

syn:
	@echo "synthesis"
	cd ${OUTPATH}/synthesis; \
        dc_shell -f main.tcl

sim:
	@echo "gate-simulation"
	cd ${OUTPATH}/testbench; \
        vcs ../verilog/define.vh tb.v ../synthesis/Netlist/${TOP}_syn.v \
            ${VLOG} \
            +define+SDF -fsdb -full64 +maxdelays

simv:
	@echo "gate-simulation"
	cd ${OUTPATH}/testbench; \
        ./simv

v2lvs:
	@echo "RO SPICE circuit"
	cd ${OUTPATH}/spice; \
        v2lvs -addpin VDD -addpin VSS -addpin VBP -addpin VBN \
            -v ../synthesis/Netlist/${TOP}_syn.v \
            -lsr merge.spx \
            -o ${TOP}.spi -i

hspice:
	@echo "SPICE simulation"
	cd ${OUTPATH}/spice; \
        hspice sim_GRO.sp -mt 20 -hpp -merge -o

hspicev:
	@echo "SPICE result"
	cd ${OUTPATH}/spice; \
        cat *.mt* |awk '/e-/{print "DL"$$3,$$1}'

