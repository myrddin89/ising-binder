#! /usr/bin/env python3

"""
    ising-binder.py - Binder cumulants for the Ising model -- v1.0

  Usage: ./ising-binder.py [-L L...] [-i INTERVAL] [-B NBETA] [-t THERMA] [-n NMC] [-s START] [-d DISTR] [-m METHOD] [-k NBOOT] [-p NPOINTS]

  Arguments:

  Options:
    -h --help    Display this help and exit
    -L L         Lattice size. This option can be repeated.
    -i INTERVAL  Interval of beta values around critical value. [default: 0.1]
    -B NBETA     Number of beta values [default: 4]
    -n NMC       Measure sweeps [default: 1000]
    -s START     Start; c for 'cold', h for 'hot' [default: c]
    -t THERMA    Therma [default: 1000]
    -k NBOOT     Number of boostrap run. [default: 5000]
    -p NPOINTS   Number of data points in bootstrap. [default: 1000]
    -d DISTR     Distribution. [default: gaus]
    -m METHOD    Method: "boot" or "jack". [default: boot]
"""

import docopt
import numpy as np
from pprint import pformat
from matplotlib import pyplot as plt
from ising import Ising
from kurt import bootstrap_jacknife


class Binder(object):
    def __init__(self, NTHERMA=1000, NMC=1000, START="c", Nboot=5000, Npoints=1000, method="boot", distr="gaus"):
        self.NTHERMA = NTHERMA
        self.NMC = NMC
        self.START = START
        self.Nboot = Nboot
        self.Npoints = Npoints
        self.method = method
        self.distr = distr
    
    def __call__(self, L, beta):
        ising = Ising(L=L, BETA=beta, NTHERMA=self.NTHERMA, NMC=self.NMC, START=self.START)
        
        ising.run()
        
        return bootstrap_jacknife(ising.ap, self.method, self.Nboot, self.Npoints, self.distr, False)



class Options(object):
    def __init__(self, argv):
        if argv["-L"]:
            self.lattices = [int(L) for L in argv["-L"]]
        else:
            self.lattices = [32, 64]
        
        self.bi    = 0.44 - float(argv["-i"])/2
        self.bf    = 0.44 + float(argv["-i"])/2
        self.Nbeta = int(argv["-B"])
        
        self.NL = len(self.lattices)
        self.NTHERMA = int(argv["-t"])
        self.NMC     = int(argv["-n"])
        self.START   = argv["-s"]
        self.method  = argv["-m"]
        self.distr   = argv["-d"]
        self.Nboot   = int(argv["-k"])
        self.Npoints = int(argv["-p"])
    
    def __str__(self):
        return pformat(self.__dict__)



def main(argv):
    options = Options(argv)
    
    betas = np.linspace(options.bi, options.bf, num=options.Nbeta, dtype=float)
    binder_list = np.zeros((options.NL, options.Nbeta), dtype=float)
    binder_err = np.zeros((options.NL, options.Nbeta), dtype=float)
    
    binder = Binder(NTHERMA=options.NTHERMA, NMC=options.NMC, START=options.START, Nboot=options.Nboot, Npoints=options.Npoints, method=options.method, distr=options.distr)
    
    for L in range(options.NL):
        for b in range(options.Nbeta):
            print("""
Computing binder for ({}, {})
            """.format(options.lattices[L], betas[b]))
            
            binder_list[L, b], binder_err[L, b] = binder(options.lattices[L], betas[b])
    
    plt.figure()
    
    for L in range(options.NL):
        plt.plot(betas, binder_list[L,:])
        
    plt.grid()
    plt.show()



if __name__ == '__main__':
    argv = docopt.docopt(__doc__, version='ising-binder.py 1.0')
    main(argv)
 
