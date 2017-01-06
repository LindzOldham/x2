import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
import pylab as pl
import numpy as np
import lensModel2


result = cPickle.load(open('/data/ljo31/Lens/J1323/emcee14'))
lp= result[0]

a1,a2,a3 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]
dx,dy,x1,y1,q1,re1,n1,pa1,q5,re5,n5,pa5,x2,y2,q2,pa2,re2,n2,q3,pa3,re3,n3,x4,y4,q4,pa4,b,eta,shear,shearpa = trace[a1,a2,a3,:]


srcs,gals,lenses = [],[],[]
print dx,dy,x1,y1,q1,re1,n1,pa1,x2,y2,q2,re2,n2,pa2,q3,pa3,re3,n3,x4,y4,q4,pa4,b,shear,shearpa
srcs.append(SBModels.Sersic('Source 1', {'x':x1,'y':y1,'q':q1,'pa':pa1,'re':re1,'n':n1}))
# if there's a second source:
srcs.append(SBModels.Sersic('Source 2', {'x':x1,'y':y1,'q':q5,'pa':pa5,'re':re5,'n':n5}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':x2,'y':y2,'q':q2,'pa':pa2,'re':re2,'n':n2}))
gals.append(SBModels.Sersic('Galaxy 2', {'x':x2,'y':y2,'q':q3,'pa':pa3,'re':re3,'n':n3}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':x4,'y':y4,'q':q4,'pa':pa4,'b':b,'eta':eta}))
lenses.append(MassModels.ExtShear('shear',{'x':x4,'y':y4,'b':shear, 'pa':shearpa}))

img1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci_cutout.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_noisemap.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf2.fits')[0].data.copy()
psf1 = psf1[10:-10,11:-10] # possibly this is too small? See how it goes
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci_cutout.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_noisemap.fits')[0].data.copy()
#psf2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf2.fits')[0].data.copy()
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

fits = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = dx,dy
        print x0,y0
    fit = lensModel2.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    fits.append(fit)

ims = []
models = []
colours = ['F606W', 'F814W']
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
    print lensModel2.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True)
    ims.append(im)
    models.append(model)
    #pl.figure()
    #pl.imshow(model[0]+model[1])
    #pl.colorbar()
    galaxy = model[0]+model[1]
    sub = image - galaxy
    #pl.figure()
    #pl.imshow(sub)
    pyfits.PrimaryHDU(sub[12:-12,12:-12]).writeto('/data/ljo31/Lens/J1323/galsub'+str(colours[i])+'.fits',clobber=True)


'''
print fits

galaxies = []
# subtract galaxy
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    fit = fits[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = dx,dy
        print x0,y0
    galaxy = fit[0]*gals[0].pixeval(xc+x0,yc+y0) + fit[1]*gals[1].pixeval(xc+x0,yc+y0)
    galaxies.append(galaxy)
    galaxy = convolve.convolve(galaxy,psf,False)[0]
    pl.figure()
    pl.imshow(galaxy)
    pl.colorbar()
    #sub = image - 
'''
'''
# plot figures
logp,coeffs,dic,vals = result
ii = np.where(logp==np.amax(logp))
coeff = coeffs[ii][0]

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
        x0,y0 = dic['xoffset'][ii][0], dic['yoffset'][ii][0]
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True)
    im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    print lensModel2.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True)
    ims.append(im)
    models.append(model)

colours = ['F606W', 'F814W']
for i in range(len(imgs)):
    image = imgs[i]
    im = ims[i]
    model = models[i]
    sigma = sigs[i]
    #pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1347/components_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1347/model_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1347/resid_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #f = open('/data/ljo31/Lens/J1347/coeff'+str(X),'wb')
    #cPickle.dump(coeff,f,2)
    #f.close()
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()

print 'source 1 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n1, '&', '%.2f'%re1, '&', '%.2f'%q1, '&','%.2f'%pa1,  r'\\'

print 'source 2 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n5, '&', '%.2f'%re5, '&', '%.2f'%q5, '&','%.2f'%pa5,  r'\\'

print 'galaxy 1 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n2, '&', '%.2f'%re2, '&', '%.2f'%q2, '&','%.2f'%pa2,  r'\\'

print 'galaxy 2 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n3, '&', '%.2f'%re3, '&', '%.2f'%q3, '&','%.2f'%pa3,  r'\\'

if trace.shape[-1] == 34:
    print 'galaxy 3 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n6, '&', '%.2f'%re6, '&', '%.2f'%q6, '&','%.2f'%pa6,  r'\\'


print 'lens 1 ', '&', '%.2f'%x4, '&',  '%.2f'%y4, '&', '%.2f'%eta, '&', '%.2f'%b, '&', '%.2f'%q4, '&','%.2f'%pa4,  r'\\\hline'

print 'shear = ', '%.4f'%shear, 'shear pa = ', '%.2f'%shearpa

'''
