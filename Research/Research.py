import numpy as np
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import gzip
import time
import math

def demo(dname, NCheb=1000, Nbin=50):
    t = time.time()
    (A,l) = load_graph(dname)
    N = nadjacency(A)
    print("Time to load and convert: %f" % (time.time() - t))
    t = time.time()
    c = moments_cheb(N,NCheb,10)
    print("Time to compute moments: %f" % (time.time() - t))
    compare_chebhist(1-l,filter_jackson(c),Nbin)

def load_graph(dname):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    smatFileName = dir_path + "\\data\\" + dname + ".smat"
    smatFileGzName = dir_path + "\\data\\" + dname + ".smat.gz"

    eigFile = dir_path + "\\data\\" + dname + ".smat.normalized.eigs"
    
    if(not os.path.isfile(eigFile)):
        raise Exception("Error: Missing eig file: %s" %eigFile)

    eigs = np.loadtxt(eigFile)
    
    if(os.path.isfile(smatFileName)):
        return (readSMAT(smatFileName),eigs)
    elif (os.path.isfile(smatFileGzName)):
        return (readSMATGZ(smatFileGzName),eigs)
    
    else:
        raise Exception ("Error: Graph %s does not exist" % dname)



def readSMAT(filename):
    print("-now reading: %s" %filename)
    if(not os.path.isfile(filename)):
        raise Exception("Error: No such file: %s" %filename)

    if(os.path.splitext(filename)[1] == ".gz"):
       m,n,i,j,v = readSMATGZ(filename)
       return sp.sparse.coo_matrix((v,(i,j)),shape=(m,n))

    if(os.path.isfile(filename+".info")):
        s = np.loadtxt(filename)
        mdata = np.loadtxt(filename+".info")
        ind_i = s[:,0]
        ind_j = s[:,1]
        val = s[:,2]

        return sp.sparse.coo_matrix((val,(ind_i,ind_j)),(mdata[0],mdata[1]))

    s = np.loadtxt(filename)

    length = s.shape[0]
    m = s[0,0]
    n = s[0,1]
    ind_i = s[1:length,0]
    ind_j = s[1:length,1]
    val = s[1:length,2]
    mat = sp.sparse.coo_matrix((val,(ind_i,ind_j)),(m,n))

    return mat

def readSMATGZ(filename):
    print("-now reading: %s" %filename)
    ## read from gz
    with gzip.open(filename, 'rb') as f:
        file_content = f.read().decode("utf-8")
        #print (file_content)
    ## write back to new file
    f = open(filename[:-3], 'w+')
    f.write(file_content)
    f.close()
    ## read that file into array
    tempArray = np.loadtxt(filename[:-3])
    ## convert array to the required format
    m = tempArray[0, 0]
    n = tempArray[0, 1]
    nnz = tempArray[0, 2]
    iarray = tempArray[1:,0]
    jarray = tempArray[1:,1]
    varray = tempArray[1:,2]
    return sp.sparse.coo_matrix((varray,(iarray,jarray)),(m,n))




def nadjacency(A):
    isSparse = sp.sparse.issparse(A)

    if(isSparse):
        [d] = np.array(coo_matrix.sum(A,1)).T
        d = d.astype(float)
        d[d!=0] = 1 / (d[d!=0] ** (0.5))
        [i,j,v] = sp.sparse.find(A)
       
        return coo_matrix((v*d[i]*d[j],(i,j)),shape=A.shape)
    else:
        d = np.array(np.sum(A,1))
        d = d.astype(float)
        d[d!=0] = 1 / (d[d!=0] ** (0.5))

        return np.dot(np.dot(np.diag(d),A),np.diag(d))


def moments_cheb(A,N=10,num_samples=100,kind=1):
    m = A.shape[0]

    Z = np.random.randn(m,num_samples)
    c = np.zeros((N,1))

    c[0] = m
    c[1] = sum(A.diagonal())
    P0 = Z
    P1 = coo_matrix.dot(A,Z) * kind

    for n in range(2,N):
        Pn = 2 * coo_matrix.dot(A,P1) - P0
        for j in range(1,num_samples):
            c[n] = c[n] + np.dot(Z[:,j] , Pn[:,j])
        c[n] = c[n] / num_samples
        P0 = P1
        P1 = Pn
    return c

def filter_jackson(c):
    c = np.array(c)
    N = c.shape[0]
    n = np.array([range(0,N)])
    tau = math.pi / (N+1)
    g = ( (N-n+1) * np.cos(tau * n) + np.sin(tau * n) / np.tan(tau) ) / (N+1)
    return g.T * c

def compare_chebhist(l,c,Nbin=25):

    lmin = max(min(l),-1)
    lmax = min(max(l),1)

    x = np.linspace(lmin,lmax,Nbin+1)
    y = plot_chebint(c,x)
    u = (x[1:] + x[:-1]) / 2
    v = (y[1:] - y[:-1])

    plt.clf()
    plt.hold(True)
    plt.hist(l,Nbin)
    plt.plot(u,v,"r.",markersize=10)
    plt.hold(False)
    plt.show()


def plot_chebint(c,xx):
    N = c.shape[0]
    txx = np.arccos(xx)
    yy = c[0] * (txx-math.pi)/2
    for NP in range(1,N):
        n = NP 
        yy = yy + c[NP] * np.sin(n*txx) / n
    yy = -2 / math.pi * yy
    return yy


demo("pgp-cc")
