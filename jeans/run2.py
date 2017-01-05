from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects

k = 4 # J0901

masses = np.load('/data/ljo31b/EELs/inference/new/masses.npy')
logM = masses[3]
dlogM = np.mean((masses[4],masses[5]),axis=0)
Mhalo = buildhalo(10**logM)
rhalo = virialRadius(Mhalo,0.5)

# construct Sersic model for source light
struct = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')[0]
phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge.fits')[1].data
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
ZPs = dict([('F555W',25.711),('F606W',26.493),('F814W',25.947)])
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
names = struct.keys()
names.sort()

# for 211 models, this is simple. 
scales = [0.05 * astCalc.da(sz[name][0]) * 1e3 * np.pi/180./3600. for name in names]
re = [struct[name]['Source 1 re'] for name in names]
n = [struct[name]['Source 1 n'] for name in names]
q = [struct[name]['Source 1 q'] for name in names]
pa = [struct[name]['Source 1 pa'] for name in names]
magv = phot['mag v']

# make spherically symmetric EELs
kpc2cm = 3.086e21
gal = SBObjects.Sersic('EEL source',{'x':0,'y':0,'re':re[k]*scales[k],'n':n[k],'q':1,'pa':0,'amp':1.})
gal.setAmpFromMag(magv[k],ZPs[bands[names[k]]])
gal.amp *= kpc2cm**2.
magsun = solarmag.getmag(bands[names[k]]+'_ACS',0.)
Lsun = -0.4*(magsun - ZPs[bands[names[k]]])
Lsun = 10**Lsun

# Mass components
r = np.logspace(-5,5,1501)
sb = gal.eval(r)
lr,light = deproject(r,sb)
Mdm = NFW(lr,rhalo[k],Mhalo[k])
Mstar = light2mass(lr,light,1.) *Lsun / kpc2cm**2.

# or
fac = Mstar[-1]/10**logM[k]
Mstar /= fac

#pl.figure()
#pl.loglog(lr,Mstar)
#pl.loglog(lr,Mdm)
#pl.xlim([1e-3,1e3])
# jeans model
sr,sigma_dmr = veldisp(r,sb,Mdm)
pl.figure()
pl.xlabel('r / kpc')
pl.ylabel('$\sigma$ / kms$^{-1}$')
pl.title(names[k])
#pl.loglog(sr,sigma_dmr**0.5)

sr,sigma_starr = veldisp(r,sb,Mstar) # need a mass-to-light ratio of about 3 to match the EELs?
#pl.loglog(sr,sigma_starr**0.5)
pl.plot(sr, (sigma_dmr+sigma_starr)**0.5)
pl.xlim([1e-2,200])
# in an aperture
sigma_dm = veldisp(r,sb,Mdm,ap=gal.re)
sigma_star = veldisp(r,sb,Mstar,ap=gal.re)
vd = (sigma_dm + sigma_star)**0.5
print vd
