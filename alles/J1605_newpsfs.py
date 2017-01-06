import pyfits as py, numpy as np, pylab as pl


I = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci.fits')[0].data.copy()

psf1=I[4653-14:4653+14,2892-14:2892+14]
psf2=I[4671-14:4671+14,2374-14:2374+14]
psf3=I[1691-14:1691+14,1974-14:1974+14]

'''pl.figure()
pl.imshow(psf1,interpolation='nearest')
pl.figure()
pl.imshow(psf2,interpolation='nearest')
pl.figure()
pl.imshow(psf3,interpolation='nearest')'''

psf1=psf1[:-1,:-2]
psf2=psf2[:-2,:-2]
psf3=psf3[:-1]

py.writeto('/data/ljo31/Lens/J1605/F814W_psf1new.fits',psf1)
py.writeto('/data/ljo31/Lens/J1605/F814W_psf2new.fits',psf2)
py.writeto('/data/ljo31/Lens/J1605/F814W_psf3new.fits',psf3)
