import numpy as np

############### returns random sample ###############
def get_vector_random(s,n):
    if (s=='gaus'):
        return np.random.normal(0,1,n)
    if (s=='unif'):
        return np.random.random(n)
    if (s=='uniuni'):
        return np.power(np.random.random(n),2)
    if (s=='dice'):
        return np.random.randint(6,size=n)
    if (s=='coin3070'):
        return np.random.binomial(1,0.3,size=n)
####################################################


def get_vector_file(pat,R,beta):
    filename=pat+'Mag_'+str(R)+'_'+str(beta)+'.dat'
    VectorRet=np.loadtxt(filename)
    Datap=len(VectorRet)
    return VectorRet, Datap
