import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

''' part one: Poisson noise '''
sci = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht.fits')[0].data.copy()

cut1 = sci[2346:2371,2082:2110] 
cut2 = sci[2224:2254,2052:2102]
cut3 = sci[2319:2365,2322:2364]

wht1 = wht[2346:2371,2082:2110] 
wht2 = wht[2224:2254,2052:2102]
wht3 = wht[2319:2365,2322:2364]


counts1 = cut1*wht1
var1 = np.var(counts1)/np.median(wht1)**2.

counts2 = cut2*wht2
var2 = np.var(counts2)/np.median(wht2)**2.

counts3 = cut3*wht3
var3 = np.var(counts3)/np.median(wht3)**2.


poisson_v = np.mean((var1,var2,var3))

sigma = poisson_v**0.5

from scipy import ndimage
im = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht_cutout_huge.fits')[0].data.copy()


smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson_v, poisson_v)**0.5
## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])


py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap)
pl.colorbar()

### I BAND

sci = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht.fits')[0].data.copy()

cut1 = sci[2346:2371,2082:2110] 
cut2 = sci[2224:2254,2052:2102]
cut3 = sci[2319:2365,2322:2364]

wht1 = wht[2346:2371,2082:2110] 
wht2 = wht[2224:2254,2052:2102]
wht3 = wht[2319:2365,2322:2364]


counts1 = cut1*wht1
var1 = np.var(counts1)/np.median(wht1)**2.

counts2 = cut2*wht2
var2 = np.var(counts2)/np.median(wht2)**2.

counts3 = cut3*wht3
var3 = np.var(counts3)/np.median(wht3)**2.


poisson_i = np.mean((var1,var2,var3))

sigma = poisson_i**0.5

from scipy import ndimage
im = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht_cutout_huge.fits')[0].data.copy()


smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson_i, poisson_i)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])

py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap)
pl.colorbar()

# note: there's this one bad pixel (far away from the lens config, so we don't actually care about it
