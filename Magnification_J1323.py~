import pylab as pl, numpy as np, pyfits
import lensModel2
from imageSim import SBModels,convolve
from pylens import *
import indexTricks as iT
import numpy

result = np.load('/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties')
lp= result[0]
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]

dx,dy,x1,y1,q1,pa1,re1,n1,q2,pa2,re2,n2,x3,y3,q3,pa3,re3,n3,x4,y4,q4,pa4,b,eta,shear,shearpa = trace[a1,a2,:]
srcs,gals,lenses = [],[],[]
srcs.append(SBModels.Sersic('Source 1', {'x':x1,'y':y1,'q':q1,'pa':pa1,'re':re1,'n':n1}))
srcs.append(SBModels.Sersic('Source 2', {'x':x1,'y':y1,'q':q2,'pa':pa2,'re':re2,'n':n2}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':x3,'y':y3,'q':q3,'pa':pa3,'re':re3,'n':n3}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':x4,'y':y4,'q':q4,'pa':pa4,'b':b,'eta':eta}))
lenses.append(MassModels.ExtShear('shear',{'x':x4,'y':y4,'b':shear, 'pa':shearpa}))


img1 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci_cutout.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_noisemap.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf.fits')[0].data.copy()
psf1 = psf1[15:-15,15:-15]
psf1 /= psf1.sum()

img2 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci_cutout.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_noisemap.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_psf_#2.fits')[0].data.copy()
psf2 = psf2[15:-15,15:-16]
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
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True)
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    ims.append(im)
    models.append(model)

### F606W ###
model = models[0]
galaxy = model[0]
source = model[1] + model[2]

print -2.5*np.log10(np.sum(source)) + 26.5


### F814 ###
model2 = models[1]
galaxy2 = model2[0]
source2 = model2[1] + model2[2]

print -2.5*np.log10(np.sum(source2)) + 25.95

galsub1 = img1 - galaxy
galsub2 = img2 - galaxy2

pyfits.writeto('/data/ljo31/Lens/J1347/galsub_F606W.fits',galsub1,clobber=True)
pyfits.writeto('/data/ljo31/Lens/J1347/galsub_F814W.fits',galsub2,clobber=True)


mask1 = pyfits.open('/data/ljo31/Lens/J1347/mask_F606W_touse.fits')[0].data.copy()
mask2 = pyfits.open('/data/ljo31/Lens/J1347/mask_F814W_touse.fits')[0].data.copy()
mask1 = mask1==1
mask2 = mask2==1

f606 = np.sum(galsub1[mask1])
f814 = np.sum(galsub2[mask2])

pl.figure()
pl.imshow(galsub1,interpolation='nearest')
pl.figure()
pl.imshow(galsub2,interpolation='nearest')

mag606 = -2.5*np.log10(f606) + 26.5
mag814 = -2.5*np.log10(f814) + 25.95

print mag606, mag814

# note, then, that there are two ways of computing the lensed image flux. (1) from the galaxy-subtracted pixels; (2) from the source model returned from lensFit. This is a good check that we're eg. masking all the galaxy in the former method. 

