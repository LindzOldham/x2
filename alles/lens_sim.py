import cPickle,numpy as np,pyfits as py,pylab as pl
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from linslens import Plotter
from tools.simple import *

# this is saying: if you have more compact things (i.e. smaller things of the same magnitude as the big things), the compact things are more likely to be lensed (i.e. more of their light gets lensed into SDSS fibre)
y,x = iT.coords((100,100))

psf = py.open('/data/ljo31/Lens/J1248/F555W_psf1.fits')[0].data.copy()
psf = psf/np.sum(psf)
psf = convolve.convolve(x,psf)[1]

rad = 30.

def sim(xs,ys,res,amp=1.):
    xl,yl = 50.1,50.1
    src = SBModels.Sersic('compact source',{'x':xl+xs,'y':yl+ys,'q':0.9,'pa':0.,'re':res,'n':4.})
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
    pl.imshow(obs,interpolation='nearest',origin='lower')
    #pl.colorbar()
    tot = np.sum(srcplane)
    imtot = np.sum(tmp)
    obstot = np.sum(obs)
    frac = obstot/tot
    print frac
    return frac

Nsamp=100
sizes = np.random.rand(Nsamp)*15. # in kpc
sizes /= (0.05*5.) # pixels
xsrc,ysrc = np.random.rand(Nsamp)*5., np.random.rand(Nsamp)*5.
frac = sizes*0.
print frac

pl.figure()
for idx in range(Nsamp):
    frac[idx] = sim(xsrc[idx],ysrc[idx],sizes[idx])
    
pl.figure()
pl.scatter(sizes*0.05*5,frac)
pl.xlabel('$r_e$ (kpc)')
pl.ylabel('fraction of light lensed into SDSS fibre')
pl.show()
