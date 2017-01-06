import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

''' part one: Poisson noise '''
sci = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F606W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F606W_wht.fits')[0].data.copy()

cut1 = sci[3638:3715,3517:3621]
cut2 = sci[4000:4045,3715:3810]
cut3 = sci[3900:3940,3910:3965]

wht1 = wht[3638:3715,3517:3621]
wht2 = wht[4000:4045,3715:3810]
wht3 = wht[3900:3940,3910:3965]

pl.figure()
pl.imshow(cut1)
pl.colorbar()
pl.figure()
pl.imshow(cut2)
pl.colorbar()
pl.figure()
pl.imshow(cut3)
pl.colorbar()

counts1 = cut1*wht1
var1 = np.var(counts1)/np.median(wht1)**2.

counts2 = cut2*wht2
var2 = np.var(counts2)/np.median(wht2)**2.

counts3 = cut3*wht3
var3 = np.var(counts3)/np.median(wht3)**2.

print var1,var2,var3
poisson = np.mean((var1,var2,var3))

sigma = poisson**0.5

from scipy import ndimage

im = py.open('/data/ljo31/Lens/J1446/F606W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1446/F606W_wht_cutout_huge.fits')[0].data.copy()

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])


py.writeto('/data/ljo31/Lens/J1446/F606W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()

### I BAND
sci = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F814W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F814W_wht.fits')[0].data.copy()

cut1 = sci[3638:3715,3517:3621]
cut2 = sci[4000:4045,3715:3810]
cut3 = sci[3900:3940,3910:3965]

wht1 = wht[3638:3715,3517:3621]
wht2 = wht[4000:4045,3715:3810]
wht3 = wht[3900:3940,3910:3965]

pl.figure()
pl.imshow(cut1)
pl.figure()
pl.imshow(cut2)
pl.figure()
pl.imshow(cut3)

counts1 = cut1*wht1
var1 = np.var(counts1)/np.median(wht1)**2.

counts2 = cut2*wht2
var2 = np.var(counts2)/np.median(wht2)**2.

counts3 = cut3*wht3
var3 = np.var(counts3)/np.median(wht3)**2.

print var1,var2,var3

poisson = np.mean((var1,var2,var3))

sigma = poisson**0.5

from scipy import ndimage
im = py.open('/data/ljo31/Lens/J1446/F814W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1446/F814W_wht_cutout_huge.fits')[0].data.copy()


smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])

## save
py.writeto('/data/ljo31/Lens/J1446/F814W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()

# note: there's this one bad pixel (far away from the lens config, so we don't actually care about it
