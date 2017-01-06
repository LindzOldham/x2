import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

im = py.open('/data/ljo31/Lens/J1125/Kp_J1125_nirc2_n.fits')[0].data.copy()

counts1 = im[500:575,500:575]
counts2 = im[1000:1055,1000:1055]
var1,var2 = np.var(counts1), np.var(counts2)
poisson = np.mean((var1,var2))
sigma = poisson**0.5

pl.figure()
pl.imshow(counts1)
pl.colorbar()
pl.figure()
pl.imshow(counts2)
pl.colorbar()

im = py.open('/data/ljo31/Lens/J1125/Kp_J1125_nirc2_n.fits')[0].data.copy()[650:905,640:915]

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im+poisson, poisson)**0.5

## get rid of nans
#ii = np.where(np.isnan(noisemap)==True)
#noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])


py.writeto('/data/ljo31/Lens/J1125/Kp_noisemap.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()
