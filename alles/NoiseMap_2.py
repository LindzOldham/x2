import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

## trying a new sort of noise map

im = py.open('/data/ljo31/Lens/SDSSJ1606+2235_F814W_sci_cutout_2.fits')[0].data
wht = py.open('/data/ljo31/Lens/SDSSJ1606+2235_F814W_wht_cutout_2.fits')[0].data
from scipy import ndimage
res = ndimage.filters.gaussian_filter(im*wht,sigma=2)

noise = 10**1.5
sn = np.where(res/noise > 7)
im2 = im*0.
im2[sn] = im[sn]
poisson = 0.0001445
im2 = im2/100. + np.sqrt(poisson)
res2 = ndimage.filters.minimum_filter(im2,5)

pl.figure()
pl.imshow(res2)
pl.colorbar()

py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F814W_noise_cutout_2.fits',res2,clobber=True)


im = py.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_sci_cutout_2.fits')[0].data
wht = py.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_wht_cutout_2.fits')[0].data
from scipy import ndimage
res = ndimage.filters.gaussian_filter(im*wht,sigma=2)

noise = 10**1.5
sn = np.where(res/noise > 7)
im2 = im*0.
im2[sn] = im[sn]
poisson = 0.0001445
im2 = im2/100. + 2.*np.sqrt(poisson)
res2 = ndimage.filters.minimum_filter(im2,5)

pl.figure()
pl.imshow(res2)
pl.colorbar()

py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F606W_noise_cutout_2.fits',res2,clobber=True)

