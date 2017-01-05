import pyfits as py, pylab as pl, numpy as np
from scipy.interpolate import splrep, splev

def clip(arr,nsig=2.5):
    a = np.sort(arr.ravel())
    a = a[a.size*0.001:a.size*0.999]
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s

spec1 = py.open('/data/ljo31b/EELs/esi/testcode/spec-4056-55357-0087.fits')[1].data
spec2 = py.open('/data/ljo31b/EELs/esi/testcode/spec-4056-55357-0119.fits')[1].data

z1,z2 = 0.50912946, 0.392347
veldisp1, veldisp2 = 181.05, 102.93


wl1,flux1 = spec1['loglam'],spec1['flux']
wl2,flux2 = spec2['loglam'],spec2['flux']
var1, var2 = spec1['ivar'], spec2['ivar']

# put onto uniform grid to add them
model = splrep(wl1,flux1)
flux1 = splev(wl2,model)
model = splrep(wl1,var1)
var1 = splev(wl2,model)

# did it work?
#pl.figure()
#pl.plot(10**wl2[10**wl2<9000],flux1[10**wl2<9000])
#pl.plot(10**wl2[10**wl2<9000],flux2[10**wl2<9000])
wl2,flux2,flux1 = wl2[10**wl2<9000], flux2[10**wl2<9000], flux1[10**wl2<9000]
var1,var2 = var1[10**wl2<9000] ,var2[10**wl2<9000]

zp1, zp2 = clip(flux1)[0], clip(flux2)[0]
# yes. Scale spectra so they're comparable in magnitude
flux1 /= zp1
flux2 /= zp2
var1 *= zp1**2.
var2 *= zp2**2.

# did it work?
pl.figure()
pl.plot(10**wl2,flux1)
pl.plot(10**wl2,flux2)

# yes. Sum spectra
#flux = flux1 * clip(flux1)[1]/clip(flux2)[0] +flux2
flux = flux1 + flux2
var = (1./var1 + 1./var2)

pl.figure()
pl.plot(10**wl2,flux)
pl.figure()
pl.plot(10**wl2[10**wl2<9000],var[10**wl2<9000])

np.save('/data/ljo31b/EELs/esi/testcode/summedspec_logSigma',[wl2,flux,var])

# resolution of sdss: ~ 2000
