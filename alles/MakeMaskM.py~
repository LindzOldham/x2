import pyfits as py, pylab as pl, numpy as np, cPickle

img1 = py.open('/data/ljo31/Lens/SDSSJ1605*3811_F555W_sci_cutout_double.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/SDSSJ1605+3811_F555W_noise_cutout_double.fits')[0].data.copy()

img2 = py.open('/data/ljo31/Lens/SDSSJ1605+3811_F814W_sci_cutout_double.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/SDSSJ1605+3811_F814W_noise_cutout_double.fits')[0].data.copy()

''' define a mask '''
import indexTricks as iT
y,x = iT.coords(img1.shape)
# centred on galaxy
xc,yc = 45,35
R = np.sqrt((x-xc)**2. + (y-yc)**2.)
# centreed on LH image
xc1,yc1 = 65,34
R1 = np.sqrt((x-xc1)**2. + (y-yc1)**2.)
# centred on RH image
xc2,yc2 = 39,37
R2 = np.sqrt((x-xc2)**2. + (y-yc2)**2.)
ii = np.where((R1>7) & (R2>3),1,0)

py.PrimaryHDU(ii).writeto('mask.fits')
