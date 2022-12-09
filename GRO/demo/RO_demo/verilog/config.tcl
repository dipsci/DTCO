# auto build by GRO compiler 09/12/2022 14:04:48
namespace eval groCfg {
    array set DC {
        libPath      ""
        libName      ""
        topModule    "GRO_TOP"
        opConditions "tt_0p45v_25c"
        clockPeriod  2.000
        drivingCell  "ND2D1LVT"
        drivingPin   "ZN"
        vlog         ""
        cload        0.0013
    }
    proc joinExt {list s} {
        set result {}
        foreach x $list {
            append result " $x$s"
        }
        return [string trim $result]
    }
}
