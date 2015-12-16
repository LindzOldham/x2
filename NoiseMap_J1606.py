import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

''' part one: Poisson noise '''
sci = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_wht.fits')[0].data.copy()

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

counts4 = cut4*wht4
var4 = np.var(counts4)/np.median(wht4)**2.

poisson = np.mean((var1,var2,var3,var4))
print poisson

sigma = poisson**0.5

from scipy import ndimage
im = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_sci_cutout_2.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_wht_cutout_2.fits')[0].data.copy()

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5
py.writeto('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_noisemap.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap)
pl.colorbar()

### I band

sci = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_sci.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_wht.fits')[0].data.copy()

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

counts4 = cut4*wht4
var4 = np.var(counts4)/np.median(wht4)**2.

poisson = np.mean((var1,var2,var3,var4))
print poisson

sigma = poisson**0.5

from scipy import ndimage
im = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_sci_cutout_2.fits')[0].data.copy()
wht = py.open('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_wht_cutout_2.fits')[0].data.copy()

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im/wht+poisson, poisson)**0.5
py.writeto('/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_noisemap.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap)
pl.colorbar()
