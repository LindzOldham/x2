import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from tools import solarmag

masses = np.load('/data/ljo31b/EELs/inference/new/masses.npy')
logM = masses[3]
dlogM = np.mean((masses[4],masses[5]),axis=0)
Mhalo = buildhalo(10**logM)
rhalo = virialRadius(Mhalo,0.5)

pl.figure()
pl.scatter(Mhalo, rhalo)

# construct Sersic model for source light
struct = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')[0]
phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge.fits')[1].data
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
ZPs = dict([('F555W',25.711),('F606W',26.493),('F814W',25.947)])
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
names = struct.keys()
names.sort()

# for 211 models, this is simple. For 212 models, we'll have to do this in 2D as we have two components. Worry about this later!
scales = [0.05 * astCalc.da(sz[name][0]) * 1e3 * np.pi/180./3600. for name in names]
re = [struct[name]['Source 1 re'] for name in names]
n = [struct[name]['Source 1 n'] for name in names]
q = [struct[name]['Source 1 q'] for name in names]
pa = [struct[name]['Source 1 pa'] for name in names]
magv = phot['mag v']

# make spherically symmetric EELs
kpc2cm = 3.086e21
gal = SBObjects.Sersic('EEL source',{'x':0,'y':0,'re':re[0]*scales[0],'n':n[0],'q':1,'pa':0,'amp':1.})
gal.setAmpFromMag(magv[0],ZPs[bands[names[0]]])
gal.amp *= kpc2cm**2.
magsun = solarmag.getmag(bands[names[0]]+'_ACS',0.)
Lsun = -0.4*(magsun - ZPs[bands[names[0]]])
Lsun = 10**Lsun

r = np.logspace(-5,5,1501)
sb = gal.eval(r)
lr,light = deproject(r,sb)
Mdm = NFW(lr,rhalo[0],Mhalo[0])
Mstar = light2mass(lr,light,3.) * kpc2cm**2. / Lsun
# this could be right! Need to check ML ratio to see if total stellar mass agrees with logM above!!!
logL = solarmag.mag_to_logL(magv[0],bands[names[0]]+'_ACS',sz[name][0])

sr,sigma = veldisp(r,sb,Mdm)
pl.figure()
pl.loglog(sr[500:],sigma[500:]**0.5)

sr,sigma = veldisp(r,sb,Mstar) # need a mass-to-light ratio of about 3 to match the EELs?
pl.loglog(sr[500:],sigma[500:]**0.5)


    
