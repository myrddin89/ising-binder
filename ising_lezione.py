#!/usr/bin/env python3

"""
    ising.py - Ising model on a square lattice in D=2, v. 1.0

  Usage: ./ising.py [-L <L>] [-b <BETA>] [-t <THERMA>] [-m <NMC>] [-s <START>] [-p <NPLOT>]

  Arguments:

  Options:
    -h --help    Display this help and exit
    -L <L>       Lattice size [default: 32]
    -b <BETA>    Inverse temperature; if negative default to critical value (see code) [default: -1.0]
    -m <NMC>     Measure sweeps [default: 1000]
    -s <START>   Start; c for 'cold', h for 'hot' [default: c]
    -t <THERMA>  Therma [default: 0]
    -p <NPLOT>   If NPLOT > 0 will plot in the end a point each NPLOT measures [default: 1]
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import docopt
import progressbar as pbar

#----------------------------------------------------------
# parse options
#----------------------------------------------------------

BETAC = math.log(1.0+math.sqrt(2.0))/2.0 # infinite volume critical value

argv = docopt.docopt(__doc__,version="1.0")

L       = int(argv["-L"])
BETA    = float(argv["-b"])
NTHERMA = int(argv["-t"])
NMC     = int(argv["-m"])
START   = argv["-s"]
NPLOT   = int(argv["-p"])
PLOTIT  = True if NPLOT > 0 else False
if BETA < 0.0:
    BETA = BETAC

V = L**2 # lattice volume
m2beta = -2.0*BETA

#----------------------------------------------------------
# define and init field; cold start by default
#----------------------------------------------------------

spin = np.ones(V, dtype=int)
if START == 'h':
    for ix in range(V):
        if random.random() < 0.5:
            spin[ix] = -1

#----------------------------------------------------------
# define and init geometry; PB Conditions
#----------------------------------------------------------

neigh = np.zeros((V,4), dtype=int)
for iy in range(L):
    for ix in range(L):
        iz = ix + iy * L
        neigh[iz,0] = iz + 1 if ix < L-1 else iz - (L-1)
        neigh[iz,1] = iz + L if iy < L-1 else iz - (L-1)*L
        neigh[iz,2] = iz - 1 if ix > 0   else iz + (L-1)
        neigh[iz,3] = iz - L if iy > 0   else iz + (L-1)*L

#----------------------------------------------------------
# a single sweep through the lattice
#----------------------------------------------------------

def sweep():
    for ix in range(V):
        sigma = spin[neigh[ix,0]] + spin[neigh[ix,1]] + spin[neigh[ix,2]] + spin[neigh[ix,3]]
        minusBetaDeltaH = m2beta*spin[ix]*sigma
        if math.log(1.0-random.random()) < minusBetaDeltaH:
            spin[ix] = -spin[ix]

#----------------------------------------------------------
# measure primary quantities
#----------------------------------------------------------

def measure():
    e = m = 0.0
    for ix in range(V):
        s = spin[ix]
        e -= s*(spin[neigh[ix,0]]+spin[neigh[ix,1]])
        m += s
    e /= V
    m /= V
    return e, m

#----------------------------------------------------------
# MAIN
#----------------------------------------------------------

bar = pbar.ProgressBar()

#----------------------------------------------------------
# therma + measure taking
#----------------------------------------------------------
namefile="Mag_"+str(L)+"_"+str(BETA)+".dat"
outfile = open(namefile,"w")

eTot = e2Tot = mTot = m2Tot = aTot = a2Tot = 0.0
ip = []; ep = []; mp = []; ap = []
j = 0
for i in bar(range(NTHERMA+NMC)):
    j += 1
    sweep()
    if j > NTHERMA:
        e, m = measure()
        a = math.fabs(m)
        ip.append(j-NTHERMA); ep.append(e); mp.append(m); ap.append(a)
        outfile.write(str(a))
        outfile.write('\n')
        eTot  += e
        mTot  += m
        aTot  += a
        e2Tot += e**2
        m2Tot += m**2
        a2Tot += a**2
        if j % NPLOT == 0:
            ip.append(j-NTHERMA); ep.append(e); mp.append(m); ap.append(a)
outfile.close()

infile=open(namefile,'r')
VVV=np.loadtxt(namefile)
print(len(VVV))

if PLOTIT:
    plt.figure(figsize=(16,9))
    plt.title(r'Ising 2$D$')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$e$, $m$, $|m|$')
    plt.plot(ip, ep, 'b-')
    plt.plot(ip, mp, 'r-')
    plt.plot(ip, ap, 'k-')
    plt.grid()
    plt.show()

eTot  /= NMC; mTot  /= NMC; aTot  /= NMC
e2Tot /= NMC; m2Tot /= NMC; a2Tot /= NMC

eErr = math.sqrt((e2Tot-eTot**2)/(NMC-1))
mErr = math.sqrt((m2Tot-mTot**2)/(NMC-1))
aErr = math.sqrt((a2Tot-aTot**2)/(NMC-1))

print("""

  < Energy > = %7.4f +/- %7.4f
  < Mag >    = %7.4f +/- %7.4f
  < |Mag| >  = %7.4f +/- %7.4f

""" % (eTot, eErr, mTot, mErr, aTot, aErr))
