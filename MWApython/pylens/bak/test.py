import numpy,pylab
from scipy import interpolate
import massmodel
import time

var = {}
const = {'x':0.,'y':0.,'q':0.3,'pa':0.,'b':1.}
M = massmodel.SIE('blah',var,const)

x0 = numpy.logspace(-3,1,41)
etas = numpy.linspace(0.,2.,21)
qs = numpy.linspace(0.2,1.,17)
grid1 = numpy.empty((x0.size,x0.size,etas.size,qs.size))
grid2 = numpy.empty(grid1.shape)
grid3 = numpy.empty((41,41))
grid4 = numpy.empty((41,41))

for i in range(qs.size):
    q = qs[i]
    if q==1.:
        q = 0.9999999
    q2 = q**2
    b = 1-q2
    s = 1e-15
    for j in range(etas.size):
        eta = etas[j]
        g = 0.5*eta-1. # g = -1*gamma
        for k in range(x0.size):
            x = x0[k]
            for l in range(x0.size):
                y = x0[l]
                print x,y,eta,q
                qb = ((2*x*y)/b)**g  # q_bar
                qt = qb*q*(x*y)**0.5/b  # q_tilde
                sb = 0.5*(x/y - y/x) + s**2*b/(2*x*y)
                nu1 = s**2*b/(2*x*y)
                nu2 = nu1+ 0.5*b*(x/y + y/(x*q2))
                nu = numpy.logspace(numpy.log10(nu1),numpy.log10(nu2),1001)
                mu = nu-sb
                t = (1+mu**2)**0.5
                f1 = (t-mu)**0.5/t
                f2 = (t+mu)**0.5/t
                ng = nu**g
                t = time.time()
                I1 = interpolate.splrep(nu,f1*ng)
                ax = qt*interpolate.splint(nu1,nu2,I1)
                print time.time()-t,ax
                from scipy import integrate
                t = time.time()
                ax = qt*integrate.trapz(f1*ng,nu)
                print time.time()-t,ax
                t = time.time()
                ax = qt*integrate.simps(f1*ng,nu)
                print time.time()-t,ax

                df
                I2 = interpolate.splrep(nu,f2*ng)
                grid1[k,l,j,i] = qt*interpolate.splint(nu1,nu2,I1)
                grid2[k,l,j,i] = qt*interpolate.splint(nu1,nu2,I2)
                #print x,y,q,g,eta,grid1[k,l,j,i],grid2[k,l,j,i]
                
import ndinterp
axes = {}
axes[0] = interpolate.splrep(x0,numpy.arange(x0.size),k=3)
axes[1] = axes[0]
axes[2] = interpolate.splrep(etas,numpy.arange(etas.size),k=1)
axes[3] = interpolate.splrep(qs,numpy.arange(qs.size),k=1)

xmodel = ndinterp.ndInterp(axes,grid1)
ymodel = ndinterp.ndInterp(axes,grid2)

import cPickle
f = open('GRIDS','wb')
cPickle.dump([xmodel,ymodel],f,2)
f.close()
