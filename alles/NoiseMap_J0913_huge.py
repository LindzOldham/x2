import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

name = 'J0913+4237'



''' part one: Poisson noise '''
sci = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F555W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F555W_wht.fits')[0].data.copy()

cut1 = sci[2410:2440,1905:1930]
cut2 = sci[2358:2380,1935:1970]
cut3 = sci[2405:2450,2050:2068]

wht1 = wht[2410:2440,1905:1930]
wht2 = wht[2358:2380,1935:1970]
wht3 = wht[2405:2450,2050:2068]

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

im = py.open('/data/ljo31/Lens/'+str(name[:5])+'/F555W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/F555W_wht_cutout_huge.fits')[0].data.copy()

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])


py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()

### I BAND
sci = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].data.copy()

cut1 = sci[2410:2440,1905:1930]
cut2 = sci[2358:2380,1935:1970]
cut3 = sci[2405:2450,2050:2068]

wht1 = wht[2410:2440,1905:1930]
wht2 = wht[2358:2380,1935:1970]
wht3 = wht[2405:2450,2050:2068]


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
im = py.open('/data/ljo31/Lens/'+str(name[:5])+'/F814W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/F814W_wht_cutout_huge.fits')[0].data.copy()


smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])

## save
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()

