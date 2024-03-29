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

def sim(xs,ys,res,comp=True,norm=False):
    compmass = (np.log10(res*0.05*5.) + 11.36)/1.05
    normmass = (np.log10(res*0.05*5.) - np.log10(2.88e-6))/0.56
    compmass = (np.log10(0.5*res*0.05*5.) - np.log10(2.88e-6))/0.56
    print compmass,normmass 
    xl,yl = 50.1,50.1
    compsrc = SBModels.Sersic('compact source',{'x':xl+xs,'y':yl+ys,'q':0.9,'pa':0.,'re':0.5*res,'n':4.,'amp':10**compmass})
    normsrc = SBModels.Sersic('compact source',{'x':xl+xs,'y':yl+ys,'q':0.9,'pa':0.,'re':res,'n':4.,'amp':10**normmass})
    if comp:
        mass=compmass
        src = compsrc
    elif norm:
        mass=normmass
        src = normsrc
    lens = MassModels.PowerLaw('lens',{'x':xl,'y':yl,'b':12.,'eta':1.,'q':0.9,'pa':90.})
    lens.setPars()
    x0,y0 = pylens.getDeflections(lens,[x,y])
    src.setPars()
    tmp = x*0
    tmp = src.pixeval(x0,y0,csub=31)
    srcplane = src.pixeval(x,y,csub=31)
    #pl.figure()
    #pl.imshow(srcplane,interpolation='nearest',origin='lower')
    # put an aperture down
    r = np.sqrt((x-xl)**2. + (y-yl)**2.)
    cond = (r<rad)
    obs = tmp.copy()
    obs[~cond] = 0
    #pl.figure()
    #pl.imshow(obs,interpolation='nearest',origin='lower')
    #pl.colorbar()
    tot = np.sum(srcplane)
    imtot = np.sum(tmp)
    obstot = np.sum(obs)
    frac = obstot/tot
    print tot, obstot,imtot
    return frac,mass

Nsamp=100
sizes = np.random.rand(Nsamp)*15. # in kpc
sizes /= (0.05*5.) # pixels
xsrc,ysrc = np.random.rand(Nsamp)*5., np.random.rand(Nsamp)*5.
compfrac = sizes*0.
normfrac = sizes*0.
compmass = sizes*0.
normmass = sizes*0.

for idx in range(Nsamp):
    compfrac[idx],compmass[idx] = sim(xsrc[idx],ysrc[idx],sizes[idx],comp=True,norm=False)
    normfrac[idx],normmass[idx] = sim(xsrc[idx],ysrc[idx],sizes[idx],comp=False,norm=True)

    
pl.figure()
pl.scatter(0.5*sizes*0.05*5,compfrac,label='compact galaxies',color='SteelBlue',s=40,alpha=0.5)
pl.scatter(sizes*0.05*5,normfrac,label='normal galaxies',color='Crimson',s=30,alpha=0.5)
pl.legend(loc='upper left')
#pl.xscale('log')
#pl.yscale('log')
pl.xlabel('$r_e$ (kpc)')
pl.ylabel('amount of light lensed into SDSS fibre')
pl.show()

pl.figure()
pl.scatter(compmass,compfrac,label='compact galaxies',color='SteelBlue',s=40,alpha=0.5)
pl.scatter(normmass,normfrac,label='normal galaxies',color='Crimson',s=30,alpha=0.5)
pl.legend(loc='upper left')
#pl.xscale('log')
#pl.yscale('log')
pl.xlabel('$\log M_{\star}$')
pl.ylabel('fraction of light lensed into SDSS fibre')
pl.show()
