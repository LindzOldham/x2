import pylab as pl, numpy as np, pyfits
import lensModel2
from imageSim import SBModels,convolve
from pylens import *
import indexTricks as iT
import numpy

result = np.load('/data/ljo31/Lens/J1323/emcee25')
lp= result[0]
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]
det = result[2]
dx,dy = det['xoffset'][a1,a2], det['yoffset'][a1,a2]

srcs,gals,lenses = [],[],[]
srcs.append(SBModels.Sersic('Source 1', {'x':det['Source 2 x'][a1,a2],'y':det['Source 2 y'][a1,a2],'q':det['Source 1 q'][a1,a2],'pa':det['Source 1 pa'][a1,a2],'re':det['Source 1 re'][a1,a2],'n':det['Source 1 n'][a1,a2]}))
srcs.append(SBModels.Sersic('Source 2', {'x':det['Source 2 x'][a1,a2],'y':det['Source 2 y'][a1,a2],'q':det['Source 2 q'][a1,a2],'pa':det['Source 2 pa'][a1,a2],'re':det['Source 2 re'][a1,a2],'n':det['Source 2 n'][a1,a2]}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':det['Galaxy 1 x'][a1,a2],'y':det['Galaxy 1 y'][a1,a2],'q':det['Galaxy 1 q'][a1,a2],'pa':det['Galaxy 1 pa'][a1,a2],'re':det['Galaxy 1 re'][a1,a2],'n':det['Galaxy 1 n'][a1,a2]}))
gals.append(SBModels.Sersic('Galaxy 2', {'x':det['Galaxy 1 x'][a1,a2],'y':det['Galaxy 1 y'][a1,a2],'q':det['Galaxy 2 q'][a1,a2],'pa':det['Galaxy 2 pa'][a1,a2],'re':det['Galaxy 2 re'][a1,a2],'n':det['Galaxy 2 n'][a1,a2]}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':det['Lens 1 x'][a1,a2],'y':det['Lens 1 y'][a1,a2],'q':det['Lens 1 q'][a1,a2],'pa':det['Lens 1 pa'][a1,a2],'b':det['Lens 1 b'][a1,a2],'eta':det['Lens 1 eta'][a1,a2]}))
lenses.append(MassModels.ExtShear('shear',{'x':det['Lens 1 x'][a1,a2],'y':det['Lens 1 y'][a1,a2],'b':det['extShear'][a1,a2], 'pa':det['extShear PA'][a1,a2]}))

img1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci_cutout.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_noisemap.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf2.fits')[0].data.copy()
psf1 = psf1[10:-10,11:-10] # possibly this is too small? See how it goes
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci_cutout.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_noisemap.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf3.fits')[0].data.copy()
psf2 = psf2[8:-8,9:-8]
psf2 /= psf2.sum()

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]


PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yc,xc = yc,xc
for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)


ims = []
models = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    print psf.shape, sigma.shape,image.shape
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = dx,dy
    print xc+x0,yc+y0
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True)
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    ims.append(im)
    models.append(model)

### F555W ###
model = models[0]
galaxy = model[0] + model[1]
source = model[2] + model[3]

mag606_1 = -2.5*np.log10(np.sum(source)) + 25.711
print -2.5*np.log10(np.sum(source)) + 25.711


### F814 ###
model2 = models[1]
galaxy2 = model2[0] + model2[1]
source2 = model2[2] + model2[3]

mag814_1 = -2.5*np.log10(np.sum(source2)) + 25.95
print -2.5*np.log10(np.sum(source2)) + 25.95

galsub1 = img1 - galaxy
galsub2 = img2 - galaxy2

pyfits.writeto('/data/ljo31/Lens/J1323/galsub_F555W.fits',galsub1,clobber=True)
pyfits.writeto('/data/ljo31/Lens/J1323/galsub_F814W.fits',galsub2,clobber=True)

'''
mask1 = pyfits.open('/data/ljo31/Lens/J1323/mask_F606W_touse.fits')[0].data.copy()
mask2 = pyfits.open('/data/ljo31/Lens/J1323/mask_F814W_touse.fits')[0].data.copy()
mask1 = mask1==1
mask2 = mask2==1

f606 = np.sum(galsub1[mask1])
f814 = np.sum(galsub2[mask2])

pl.figure()
pl.imshow(galsub1,interpolation='nearest')
pl.figure()
pl.imshow(galsub2,interpolation='nearest')

mag606 = -2.5*np.log10(f606) + 25.711
mag814 = -2.5*np.log10(f814) + 25.95

print mag606, mag814

# note, then, that there are two ways of computing the lensed image flux. (1) from the galaxy-subtracted pixels; (2) from the source model returned from lensFit. This is a good check that we're eg. masking all the galaxy in the former method. 


'''
