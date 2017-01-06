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
masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_212.npy')
logM = masses[3]

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
names = sz.keys()
names.sort()
names = np.delete(names,6)
dir = '/data/ljo31/Lens/LensModels/twoband/'
from linslens import EELsModels_huge as L
allfluxes,real_res  = [], []
res = np.linspace(1.,100.,100)
ii=0
for name in names:
    fluxes = []
    try:
        result = np.load(dir+name+'_212')
    except:
        result = np.load(dir+name+'_112')
    model = L.EELs(result, name)
    model.Initialise()
    mag = model.GetIntrinsicMags()[0]
    re_true = model.GetSourceSize(kpc=True)[0]
    print re_true
    lenses,srcs = model.lenses, model.srcs
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.getDeflections(lenses,[x,y])
    xl,yl = lenses[0].x, lenses[0].y
    r = np.sqrt((x-xl)**2. + (y-yl)**2.)
    cond = (r<rad)
    srcs = model.srcs
    for src in srcs:
        src.setPars()
    # real:
    src.setAmpFromMag(mag,48.10)
    #print src.amp, src.re
    tmp = src.pixeval(x0,y0,csub=31)
    tmp = convolve.convolve(tmp,psf,False)[0]
    obs = tmp.copy()
    obs[~cond] = 0
    norm = np.sum(obs)
    # what would the size be, given the stellar mass, for Shen relation and half Shen's relation?
    logre1 = np.log10(2.88e-6) + 0.56*logM[ii] 
    re1, re2 = 10**logre1 / 0.05/model.scale, 0.5*10**logre1 / 0.05/model.scale
    for re in res:
        tmp = x*0.
        src.re = re
        src.setAmpFromMag(mag,48.10)
        #print src.amp, src.re
        tmp = src.pixeval(x0,y0,csub=31)
        tmp = convolve.convolve(tmp,psf,False)[0]
        obs = tmp.copy()
        obs[~cond] = 0
        fluxes.append(np.sum(obs)/norm)
    allfluxes.append(fluxes)
    real_res.append(re_true)
    #pl.figure()
    #pl.plot(res,fluxes/np.max(fluxes))
    #pl.title(name)
    #pl.axvline(re_true,color='k')
    #pl.axvline(re1,color='k',ls='--')
    #pl.axvline(re2,color='k',ls='--')
    #pl.xlabel('$r_e$ (pixels)')
    #pl.ylabel('flux (image units)')
    #pl.savefig('/data/ljo31/public_html/Lens/phys_models/'+name+'_flux.png')
    #pl.show()
    ii+=1

allfluxes = np.array(allfluxes)
# now make composite flux curve to use in inference
f = res*0.
for ii in range(res.size):
    f[ii] = np.median(allfluxes[:,ii])

np.savetxt('/data/ljo31b/EELs/sizemass/re_flux.dat',np.column_stack((res,f/np.max(f))))
