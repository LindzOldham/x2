import pyfits as py, numpy as np, pylab as pl
import indexTricks as iT
from scipy import ndimage

name = 'J0837+0801'

im = py.open('/data/ljo31/Lens/'+str(name[:5])+'/J0837_Kp_narrow_med.fits')[0].data.copy()

counts1 = im[960:1080,330:480]
counts2 = im[600:700,1080:1300]
var1,var2 = np.var(counts1), np.var(counts2)
poisson = np.mean((var1,var2))
sigma = poisson**0.5



im = py.open('/data/ljo31/Lens/'+str(name[:5])+'/J0837_Kp_narrow_med.fits')[0].data.copy()[810:1100,790:1105]

smooth = ndimage.gaussian_filter(im,0.7)
noisemap = np.where((smooth>0.7*sigma)&(im>0),im+poisson, poisson)**0.5

## get rid of nans
ii = np.where(np.isnan(noisemap)==True)
noisemap[ii] = np.amax(noisemap[np.isnan(noisemap)==False])


py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/Kp_noisemap.fits',noisemap,clobber=True)
pl.figure()
pl.imshow(noisemap,origin='lower',interpolation='nearest')
pl.colorbar()
