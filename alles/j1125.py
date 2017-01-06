import pyfits as py, pylab as pl, numpy as np
from linslens import EELsLensModels as L
from scipy.interpolate import splrep, splev, splint
import lenslib

result = np.load('/data/ljo31/Lens/LensModels/J1125_212_nonconcentric')
model = L.EELs(result,name='J1125')
model.Initialise()
gals = model.gals
fits = model.fits
Xgrid = np.logspace(-4,5,1501)
galaxy = fits[0][0]*gals[0].eval(Xgrid) + fits[0][1]*gals[1].eval(Xgrid)
R = Xgrid.copy()
light = galaxy*2.*np.pi*R
mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
intlight = splint(0,model.lenses[0].pars['b'],mod)
totlight = splint(0,R[-1],mod)
frac =  intlight/totlight
print frac
mass = np.log10(lenslib.sie_mass(17*0.05,0.442,0.6885))
totmass = mass - np.log10(frac)
print totmass

## let's also do the same for our 212 concentric  model
result = np.load('/data/ljo31/Lens/LensModels/J1125_212_concentric')
model = L.EELs(result,name='J1125')
model.Initialise()
gals = model.gals
fits = model.fits
Xgrid = np.logspace(-4,5,1501)
galaxy = fits[0][0]*gals[0].eval(Xgrid) + fits[0][1]*gals[1].eval(Xgrid)
R = Xgrid.copy()
light = galaxy*2.*np.pi*R
mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
intlight = splint(0,model.lenses[0].pars['b'],mod)
totlight = splint(0,R[-1],mod)
frac =  intlight/totlight
print frac
mass = np.log10(lenslib.sie_mass(17*0.05,0.442,0.6885))
totmass = mass - np.log10(frac)
print totmass

###
result = np.load('/data/ljo31/Lens/LensModels/J1125_211')
model = L.EELs(result,name='J1125')
model.Initialise()
gals = model.gals
fits = model.fits
Xgrid = np.logspace(-4,5,1501)
galaxy = fits[0][0]*gals[0].eval(Xgrid) + fits[0][1]*gals[1].eval(Xgrid)
R = Xgrid.copy()
light = galaxy*2.*np.pi*R
mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
intlight = splint(0,model.lenses[0].pars['b'],mod)
totlight = splint(0,R[-1],mod)
frac =  intlight/totlight
print frac
mass = np.log10(lenslib.sie_mass(17*0.05,0.442,0.6885))
totmass = mass - np.log10(frac)
print totmass
