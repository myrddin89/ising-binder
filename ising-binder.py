#! /usr/bin/env python3

import progressbar as pbar
import numpy as np
from matplotlib import pyplot as plt
from ising import Ising
from kurt import bootstrap_jacknife

class Binder(object):
    def __init__(self, NTHERMA=1000, NMC=1000, START="c", Nboot=5000, Npoints=1000, method="boot", distr="gaus"):
        self.NTHERMA = 10000
        self.NMC = 91000
        self.START = "c"
        self.Nboot = 5000
        self.Npoints = 1000
        self.method="boot"
        self.distr = "gaus"

    def __call__(self, L, beta):
        ising = Ising(L=L, BETA=beta, NTHERMA=self.NTHERMA, NMC=self.NMC, START=self.START)

        ising.run()

        return bootstrap_jacknife(ising.ap, self.method, self.Nboot, self.Npoints, self.distr, False)

def main():
    bi = 0.43
    bf = 0.45
    lattices = [32, 64]
    NL = len(lattices)
    Nbeta = 10
    NTHERMA = 1000
    NMC = 1000
    START = "c"
    Nboot = 5000
    Npoints = 1000
    method = "boot"
    distr = "gaus"

    betas = np.linspace(bi, bf, num=Nbeta, dtype=float)
    binder_list = np.zeros((NL, Nbeta), dtype=float)
    binder_err = np.zeros((NL, Nbeta), dtype=float)

    binder = Binder(NTHERMA=NTHERMA, NMC=NMC, START=START, Nboot=Nboot, Npoints=Npoints, method=method, distr=distr)

    # barL = pbar.ProgressBar()
    # barB = pbar.ProgressBar()

    for L in range(NL):
        for b in range(Nbeta):
            print("""
Computing binder for ({}, {})
            """.format(lattices[L], betas[b]))
            binder_list[L, b], binder_err[L, b] = binder(lattices[L], betas[b])

    plt.figure()

    for L in range(NL):
        plt.plot(betas, binder_list[L,:])

    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
