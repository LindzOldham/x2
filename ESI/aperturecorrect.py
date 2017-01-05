import pyfits as py, pylab as pl, numpy as np
import cPickle
from astLib import astCalc

def clip(arr,nsig=3.):
    a = arr.flatten()
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<s*nsig]
        if a.size==l:
            return m,s

def apcorr(ap,re):
    log = np.log10(ap/re)
    return -0.065*log -0.013*log**2.

table = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
names,res = table['name'],table['re v']
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
apcor = []

for n in range(names.size):
    name,re = names[n],res[n]
    # get aperture we extracted spectrum over
    aps = np.load('/data/ljo31b/EELs/esi/kinematics/'+name+'_apertures.npy')
    ap = clip(aps)[0]/2.
    #print '######## \n', name
    #print 'aperture size in arcsec: ', '%.4f'%ap
    # get effective radius of source in arcsec - have to convert from kpc
    z = sz[name][0]
    #print 'redshift: ', '%.4f'%z
    Da = astCalc.da(z)
    scale = Da*1e3*np.pi/180./3600.
    re /= scale
    #print 'effective radius in arcsec: ', '%.4f'%re
    #print 'aperture correction should be ', '%.4f'%(10**apcorr(ap,re))
    print name, ' & ', '%.4f'%ap, ' & ',  '%.4f'%re, ' & ', '%.4f'%(10**apcorr(ap,re)), r'\\'
    apcor.append(10**apcorr(ap,re))

np.save('/data/ljo31b/EELs/esi/kinematics/aperture_corrections',apcor)

