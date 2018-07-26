import math
import random
import numpy as np
import progressbar as pbar

BETAC = math.log(1.0+math.sqrt(2.0))/2.0 # infinite volume critical value

class Ising(object):
    def __init__(self, L=32, BETA=BETAC, NTHERMA=1000, NMC=2000, START='c'):
        self.L = L
        self.BETA = BETA
        self.NTHERMA = NTHERMA
        self.NMC = NMC
        self.START = START

        if self.BETA < 0.0:
            self.BETA = self.BETAC

        self.V = self.L**2 # lattice volume
        self.m2beta = -2.0*self.BETA

        self.spin = np.ones(self.V, dtype=int)
        if self.START == 'h':
            for ix in range(self.V):
                if random.random() < 0.5:
                    self.spin[ix] = -1

        self.neigh = np.zeros((self.V,4), dtype=int)
        for iy in range(self.L):
            for ix in range(self.L):
                iz = ix + iy * self.L
                self.neigh[iz,0] = iz + 1 if ix < self.L-1 else iz - (self.L-1)

                self.neigh[iz,1] = iz + self.L if iy < self.L-1 else iz - (self.L-1)*self.L
                self.neigh[iz,2] = iz - 1 if ix > 0   else iz + (self.L-1)
                self.neigh[iz,3] = iz - self.L if iy > 0   else iz + (self.L-1)*self.L

    def sweep(self):
        for ix in range(self.V):
            sigma = self.spin[self.neigh[ix,0]] + self.spin[self.neigh[ix,1]] + self.spin[self.neigh[ix,2]] + self.spin[self.neigh[ix,3]]
            minusBetaDeltaH = self.m2beta*self.spin[ix]*sigma
            if math.log(1.0-random.random()) < minusBetaDeltaH:
                self.spin[ix] = -self.spin[ix]

    def measure(self):
        e = m = 0.0
        for ix in range(self.V):
            s = self.spin[ix]
            e -= s*(self.spin[self.neigh[ix,0]]+self.spin[self.neigh[ix,1]])
            m += s
        e /= self.V
        m /= self.V
        return e, m

    def run(self):
        bar = pbar.ProgressBar()

        #----------------------------------------------------------
        # therma + measure taking
        #----------------------------------------------------------
        namefile="Mag_"+str(self.L)+"_"+str(self.BETA)+".dat"
        outfile = open(namefile,"w")

        eTot = e2Tot = mTot = m2Tot = aTot = a2Tot = 0.0
        ip = []; ep = []; mp = []; ap = []
        j = 0
        for i in bar(range(self.NTHERMA+self.NMC)):
            j += 1
            self.sweep()
            if j > self.NTHERMA:
                e, m = self.measure()
                a = math.fabs(m)
                ip.append(j-self.NTHERMA); ep.append(e); mp.append(m); ap.append(a)
                outfile.write(str(a))
                outfile.write('\n')
                eTot  += e
                mTot  += m
                aTot  += a
                e2Tot += e**2
                m2Tot += m**2
                a2Tot += a**2

        outfile.close()

        eTot  /= self.NMC; mTot  /= self.NMC; aTot  /= self.NMC
        e2Tot /= self.NMC; m2Tot /= self.NMC; a2Tot /= self.NMC

        eErr = math.sqrt((e2Tot-eTot**2)/(self.NMC-1))
        mErr = math.sqrt((m2Tot-mTot**2)/(self.NMC-1))
        aErr = math.sqrt((a2Tot-aTot**2)/(self.NMC-1))

        print("""

          < Energy > = %7.4f +/- %7.4f
          < Mag >    = %7.4f +/- %7.4f
          < |Mag| >  = %7.4f +/- %7.4f

        """ % (eTot, eErr, mTot, mErr, aTot, aErr))

#----------------------------------------------------------
# MAIN
#----------------------------------------------------------


# infile=open(namefile,'r')
# VVV=np.loadtxt(namefile)
# print(len(VVV))
#
# if PLOTIT:
#     plt.figure(figsize=(16,9))
#     plt.title(r'Ising 2$D$')
#     plt.xlabel(r'$i$')
#     plt.ylabel(r'$e$, $m$, $|m|$')
#     plt.plot(ip, ep, 'b-')
#     plt.plot(ip, mp, 'r-')
#     plt.plot(ip, ap, 'k-')
#     plt.grid()
#     #plt.show()
