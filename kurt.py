#!/usr/bin/env python3

"""
    kurt_bootstrap.py

  Usage: ./ising.py [-M <method>] [-I <NN>] [-N <Nboot>] -P <datapoints> [-d <distr>] [--dontsave] <file>

  Arguments:
    <file>            Data filename

  Options:
    -h --help         Display this help and exit
    --dontsave        Do not save the plots
    -M <method>       Method: boot/jack [default: boot]
    -I <NN>           Repear the experiment NN times [default: 1]
    -N <Nboot>        Bootstrap sample dimension. Use 0 to work only on sample [default: 5000]
    -P <datapoints>   Number of data points
    -d <distr>        Distribution (gaus/unif/uniuni/dice/coin3070) [default: gaus]


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import moment
import docopt
import progressbar as pbar
import get_val
import sys

def bootstrap_jacknife(data, T, Nboot, datapoints, distr, want_vector):
    #VV, lllll=get_val.get_vector_file('../lezioni/',32,0.44)

    VV = data.copy()

    v=[]
    
    if T == "boot":
        corr_fac=1
    else:
        corr_fac=np.sqrt(datapoints-1)
        
    if (Nboot==0):
        v.append((moment(VV,4)/moment(VV,2)**2)-3)

    for k in range(Nboot):
        if (k%1000==0):
            print((Nboot-k)/1000)

        if (T=="boot"):
            indici=np.random.randint(datapoints,size=datapoints)
            boots=[VV[z] for z in indici]


        elif (T=="jack"):
            boots=np.delete(VV,k)


        curt=(moment(boots,4)/moment(boots,2)**2)-3

        v.append(curt)

    if (want_vector):
        return v, ((moment(VV,4)/moment(VV,2)**2)-3), np.mean(v), np.std(v)*corr_fac
    else:
        return np.mean(v), np.std(v)*corr_fac


if __name__ == "__main__":
    X=[]
    S=[]
    c05=0
    c1=0
    c2=0
    c3=0
    quantile=[]
    Teo={'gaus':0., 'unif':(-6./5), 'uniuni':(-6./7), 'dice':(-1563./1225), 'coin3070':(-26./21)}


    ############# arguments ####################
    argv = docopt.docopt(__doc__,version="1.0")
    NN=int(argv["-I"])    #repeat the program NN times --> get NN bootstram distributions
    Nboot=int(argv["-N"])   #bootstrap sample dimension (on datapoints)
    datapoints=int(argv["-P"])
    distr=str(argv["-d"])
    method=str(argv["-M"])  # jack or boot
    saveimg=not bool(argv["--dontsave"])
    filename=argv["<file>"]
    ###########################################


    ########## settings depending on arguments ##########
    teo=Teo[distr]
    if (Nboot==0):
        method='boot'
    if (method=="boot"):
        namefig='images/'+str(NN)+'volte_'+str(Nboot)+'bootstrap_su'+str(datapoints)+'dati_'+distr+'.png'
        corr_fac=1
        TT='Bootstrap'
    elif (method=="jack"):
        namefig='images/'+str(NN)+'volte_'+str(datapoints)+'dati_jacknife'+distr+'.png'
        corr_fac=np.sqrt(datapoints-1)
        Nboot=datapoints
        TT='Jacknife'
    if (Nboot==0):
        namefig='images/Real_distribution_of_'+str(NN)+'_kurtosis_on_'+str(datapoints)+'_data_'+distr+'.png'
    #####################################################


    ############### returns random sample ###############
    #def get_vector(s,n):
    #    if (s=='gaus'):
    #        return np.random.normal(0,1,n)
    #    if (s=='unif'):
    #        return np.random.random(n)
    #    if (s=='uniuni'):
    #        return np.power(np.random.random(n),2)
    #    if (s=='dice'):
    #        return np.random.randint(6,size=n)
    #    if (s=='coin3070'):
    #        return np.random.binomial(1,0.3,size=n)
    #####################################################

    #VV, lllll=get_val.get_vector_file('../lezioni/',32,0.44)

    VV, lllll = get_val.get_vector_file(filename)

    measured=-999.
    for j in range(NN):
        print("----->j=",j)


        #### plotting gaussian only if j=0 & Nboot>0
        if (j==0 and Nboot>0):
            v,measured,x,s=bootstrap_jacknife(VV, method,Nboot,datapoints,distr,True)
            s=s/corr_fac
            nbin=int(np.sqrt(Nboot))
            if (NN>9):
                plt.subplot(2,1,1)
            plttitle=TT+' distribution '+str(NN)+' '+str(Nboot)+' '+str(datapoints)+' '+distr
            plt.title(plttitle)
            plt.hist(v,nbin,normed=True)
            minbin=x-4*s
            maxbin=x+4*s
            bins=[minbin+(maxbin-minbin)*ii/nbin for ii in range(nbin)]
            plt.plot(bins,1/(s*np.sqrt(2*np.pi))*np.exp(-(bins-x)**2/(2*s**2)),linewidth=2,color='r')
            fx, fy = [teo,teo], [0,1.05/(s*np.sqrt(2*np.pi))]
            if not(method=="jack" and NN>9):
                plt.plot(fx,fy,color='g')
            textt='Misurato: %7.4f \n%s: %7.4f +/- %7.4f'%(measured,TT,x,s*corr_fac)
            plt.text(x-3.*s,1./(s*np.sqrt(2*np.pi)),textt)
            s=s*corr_fac
        else:
            x,s=bootstrap_jacknife(VV, method,Nboot,datapoints,distr,False)
        X.append(x)
        S.append(s)
        quant=x
        if (Nboot>0):
            quant=abs(quant-teo)/s
        quantile.append(quant)
        c05+=(quant<0.5); c1+=(quant<1);
        c2+=(quant<2); c3+=(quant<3)

        print("In s/2: ",100.*c05/(j+1),"%\nIn s: ",100.*c1/(j+1),"%\nIn 2s:",100.*c2/(j+1),"%\nIn 3s:",100.*c3/(j+1),"%")





    #####plotting quantiles
    if (NN>9):
        if (Nboot>0):
            plt.subplot(2,1,2)
        summary='Teorico: %7.4f \nMisurato: %7.4f \nSingolo esperimento: %7.4f +/- %7.4f \nMedia delle medie %s: %7.4f \nRMS delle medie %s: %7.4f \nMedia delle varianze %s: %7.4f \nRMS delle varianze %s: %7.4f'%(teo,measured,X[0],S[0],TT,np.mean(X),TT,np.std(X),TT,np.mean(S),TT,np.std(S))
        if (Nboot>0):
            plt.text(1.5,0.25,summary)
        else:
            summary=summary+'\n(Ignora \"MISURATO\" e \"SINGOLO ESP.\")'
            plt.text(teo,1,summary)
        plt.hist(quantile,int(np.sqrt(NN)),normed=True,color='g')
        xq=[k/100. for k in range(400)]
        yq=[2/(np.sqrt(2*np.pi))*np.exp(-(QQ)**2/2) for QQ in xq]
        if (Nboot>0):
            plt.plot(xq,yq,linewidth=2,color='b')


    if (saveimg):
        plt.savefig(namefig)
    else:
        plt.show()
