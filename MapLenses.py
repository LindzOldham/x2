from pylens import MassModels,pylens
from scipy.special import gamma,gammainc
import itertools
import numpy as np, pylab as pl

sersic = lambda n,Re,k,R:  np.exp(-k*((R/Re)**(1./n))) * (k**(2.*n)) / (2.*np.pi*n*gamma(2.*n)*  gammainc(2.*n,k*(240./Re)**(1./n))  * Re**2)

x,y=np.linspace(-20,20,700),np.linspace(-20,20,700)
ii = np.array(list(itertools.product(x,y)))
lens = MassModels.PowerLaw('Lens1', {'x':0,'y':0,'q':1,'pa':0,'b':10,'eta':0.7})
xd,yd = pylens.getDeflections(lens,ii.T)
rl = np.sqrt(xd**2. + yd**2.)
SB = sersic(4,30,(2.*4-0.324), rl)

#pl.figure()
#pl.scatter(ii[:,0],ii[:,1],c=SB,edgecolors='none',s=100)
#pl.figure()
#pl.scatter(xd,yd,c=SB,edgecolors='none',s=100)

# caustic?
# source relative to galaxy?

DXs,DYs = np.linspace(0.1,1,3), np.linspace(0.1,1,3)
jj = np.array(list(itertools.product(DXs,DYs)))
for i in range(jj.shape[0]):
    DX,DY=jj[i]
    rl = np.sqrt((xd-DX)**2. + (yd-DY)**2.)
    SB = sersic(4,30,(2.*4-0.324), rl)
    pl.figure()
    pl.scatter(ii[:,0],ii[:,1],c=SB,edgecolors='none',s=100)
    pl.scatter(0,0,color='White')
    pl.scatter(DX,DY,color='Yellow')
    pl.axis([-20,20,-20,20])
