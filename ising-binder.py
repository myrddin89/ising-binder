#! /usr/bin/env python3

import numpy as np
from ising import Ising
from kurt import bootstrap_jacknife

def main():
    ising1 = Ising(L=32, BETA=0.43, NTHERMA=1000, NMC=2000, START="c")
    ising2 = Ising(L=32, BETA=0.45, NTHERMA=1000, NMC=2000, START="c")

    ising1.run()
    ising2.run()

    Nboot = 5000
    datapoints = 1000

    Ul1, err1 = bootstrap_jacknife(ising1.ap, "boot", Nboot, datapoints, "gaus", False)

    Ul2, err2 = bootstrap_jacknife(ising2.ap, "boot", Nboot, datapoints, "gaus", False)

    print("""\
    Binder1:
        U_L = {:7.4f} +/- {:7.4f}
    Binder2:
        U_L = {:7.4f} +/- {:7.4f}
    """.format(Ul1, err1, Ul2, err2))

if __name__ == '__main__':
    main()
