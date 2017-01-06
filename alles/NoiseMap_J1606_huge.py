import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

name = 'J1606+2235'

''' part one: Poisson noise '''
sci = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].data.copy()

cut1 = sci[2040:2080,2170:2210]
cut2 = sci[2040:2120,2130:2210]
cut3 = sci[2180:2250,2200:2280]
cut4 = sci[1850:1925,2345:2420]

wht1 = wht[2040:2080,2170:2210]
wht2 = wht[2040:2120,2130:2210]
wht3 = wht[2180:2250,2200:2280]
wht4 = wht[1850:1925,2345:2420]


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

im = py.open('/data/ljo31/Lens/'+str(name[:5])+'/F606W_sci_cutout_huge.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/F606W_wht_cutout_huge.fits')[0].data.copy()

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])


py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_noisemap_huge.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()

### I BAND
sci = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].data.copy()

cut1 = sci[2040:2080,2170:2210]
cut2 = sci[2040:2120,2130:2210]
cut3 = sci[2180:2250,2200:2280]
cut4 = sci[1850:1925,2345:2420]

wht1 = wht[2040:2080,2170:2210]
wht2 = wht[2040:2120,2130:2210]
wht3 = wht[2180:2250,2200:2280]
wht4 = wht[1850:1925,2345:2420]


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

