import cPickle,numpy as np,pyfits as py,pylab as pl
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from linslens import Plotter
from tools.simple import *

y,x = iT.coords((100,100))

psf = py.open('/data/ljo31/Lens/J1248/F555W_psf1.fits')[0].data.copy()
psf = psf/np.sum(psf)
psf = convolve.convolve(x,psf)[1]

rad = 30.

def sim(xs,ys,mass,comp=True,norm=False):
    xl,yl = 50.1,50.1
    logrenorm = np.log10(2.88e-6) + 0.56*mass
    renorm = 10**logrenorm
    recomp = 0.5*renorm
    normsrc = SBModels.Sersic('compact source',{'x':xl+xs,'y':yl+ys,'q':0.9,'pa':0.,'re':renorm/0.05/5.,'n':4.,'amp':10**mass})
    compsrc = SBModels.Sersic('compact source',{'x':xl+xs,'y':yl+ys,'q':0.9,'pa':0.,'re':recomp/0.05/5.,'n':4.,'amp':10**mass})
    if comp:
        src = compsrc
    elif norm:
        src = normsrc
    lens = MassModels.PowerLaw('lens',{'x':xl,'y':yl,'b':12.,'eta':1.,'q':0.9,'pa':90.})
    lens.setPars()
    x0,y0 = pylens.getDeflections(lens,[x,y])
    src.setPars()
    tmp = x*0
    tmp = src.pixeval(x0,y0,csub=31)
    srcplane = src.pixeval(x,y,csub=31)
    r = np.sqrt((x-xl)**2. + (y-yl)**2.)
    cond = (r<rad)
    obs = tmp.copy()
    obs[~cond] = 0
    tot = np.sum(srcplane)
    imtot = np.sum(tmp)
    obstot = np.sum(obs)
    frac = obstot/tot
    return frac,renorm

Nsamp=500
masses = np.random.rand(Nsamp)*1.5+10. # in kpc
sizes = np.random.rand(Nsamp)*15.
sizes /= (0.05*5.) # pixels
xsrc,ysrc = np.random.rand(Nsamp)*5., np.random.rand(Nsamp)*5.
compfrac = sizes*0.
normfrac = sizes*0.
compre = sizes*0.
normre = sizes*0.

for idx in range(Nsamp):
    compfrac[idx],normre[idx] = sim(xsrc[idx],ysrc[idx],masses[idx],comp=True,norm=False)
    normfrac[idx],normre[idx] = sim(xsrc[idx],ysrc[idx],masses[idx],comp=False,norm=True)
compre = normre*0.5
    


pl.figure()
pl.scatter(masses,compfrac,s=100,edgecolors='none',color='SteelBlue',alpha=0.5,label='compact galaxies')
pl.scatter(masses,normfrac,s=100,edgecolors='none',color='Crimson',alpha=0.5,label='normal galaxies')
pl.legend(loc='upper left')
#pl.xscale('log')
#pl.yscale('log')
pl.xlabel('$\log M_{\star}$')
pl.ylabel('fraction of light lensed into SDSS fibre')
pl.show()
