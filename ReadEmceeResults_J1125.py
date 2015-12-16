import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
import pylab as pl
import numpy as np
import lensModel2
from matplotlib.colors import LogNorm

def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=0,vmax=1) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    #pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-3,vmax=3,cmap='afmhot',aspect='auto')
    #pl.colorbar()
    #pl.title('signal-to-noise residuals 1')
    pl.imshow((image-im),origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    #pl.colorbar()
    #pl.title('signal-to-noise residuals 1')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals 2')
    pl.colorbar()

file = '/data/ljo31/Lens/J1125/emcee13'
result = np.load(file)
print file

lp= result[0]
#ii = np.where(lp>-23393)
#lp2 = np.delete(lp,ii,axis=1)
#jj = np.where(lp2>-23429)
#lp2 = np.delete(lp2,jj,axis=1)

#a1,a2 = numpy.unravel_index(lp2.argmax(),lp2.shape)
#a1,a2 = np.array(np.where(lp==lp2[a1,a2])).T[0]
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]
dic = result[2]

dx,dy = dic['xoffset'][a1,a2], dic['yoffset'][a1,a2]
x1,y1,re1,n1,pa1,q1 = dic['Source 1 x'][a1,a2], dic['Source 1 y'][a1,a2], dic['Source 1 re'][a1,a2], dic['Source 1 n'][a1,a2], dic['Source 1 pa'][a1,a2], dic['Source 1 q'][a1,a2]
#re5,n5,pa5,q5 = dic['Source 2 re'][a1,a2], dic['Source 2 n'][a1,a2], dic['Source 2 pa'][a1,a2], dic['Source 2 q'][a1,a2]
x2,y2,re2,n2,pa2,q2 = dic['Galaxy 1 x'][a1,a2], dic['Galaxy 1 y'][a1,a2], dic['Galaxy 1 re'][a1,a2], dic['Galaxy 1 n'][a1,a2], dic['Galaxy 1 pa'][a1,a2], dic['Galaxy 1 q'][a1,a2]
#re3,n3,pa3,q3 = dic['Galaxy 2 re'][a1,a2], dic['Galaxy 2 n'][a1,a2], dic['Galaxy 2 pa'][a1,a2], dic['Galaxy 2 q'][a1,a2]
#re6,n6,pa6,q6 = dic['Galaxy 3 re'][a1,a2], dic['Galaxy 3 n'][a1,a2], dic['Galaxy 3 pa'][a1,a2], dic['Galaxy 3 q'][a1,a2]
x4,y4,b,eta,pa4,q4 = dic['Lens 1 x'][a1,a2], dic['Lens 1 y'][a1,a2], dic['Lens 1 b'][a1,a2], dic['Lens 1 eta'][a1,a2], dic['Lens 1 pa'][a1,a2], dic['Lens 1 q'][a1,a2]
shear,shearpa = dic['extShear'][a1,a2], dic['extShear PA'][a1,a2]

x1,y1 = x1+x4, y1+y4
srcs,gals,lenses = [],[],[]
srcs.append(SBModels.Sersic('Source 1', {'x':x1,'y':y1,'q':q1,'pa':pa1,'re':re1,'n':n1}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':x2,'y':y2,'q':q2,'pa':pa2,'re':re2,'n':n2}))
#gals.append(SBModels.Sersic('Galaxy 2', {'x':x2,'y':y2,'q':q3,'pa':pa3,'re':re3,'n':n3}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':x4,'y':y4,'q':q4,'pa':pa4,'b':b,'eta':eta}))
lenses.append(MassModels.ExtShear('shear',{'x':x4,'y':y4,'b':shear, 'pa':shearpa}))

img1 = py.open('/data/ljo31/Lens/J1125/F606W_sci_cutout.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1125/F606W_noisemap.fits')[0].data.copy()
psf1 = py.open('/data/ljo31/Lens/J1125/F606W_psf1.fits')[0].data.copy()
psf1 = psf1[5:-5,5:-6]
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1125/F814W_sci_cutout.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1125/F814W_noisemap.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1125/F814W_psf2.fits')[0].data.copy()
psf2 = psf2[7:-6,7:-8]
psf2 = psf2/np.sum(psf2)

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



lp = 0.
for i in range(len(imgs)):
        if i == 0:
            x0,y0 = 0,0
        else:
            x0 = dx
            y0 = dy
            print x0,y0
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        lp += lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,verbose=False,psf=psf,csub=1)

print 'lp = ', lp
# lp = -8686

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
    py.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1125/components_'+str(colours[i])+'.fits',clobber=True)
    py.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1125/model_'+str(colours[i])+'.fits',clobber=True)
    py.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1125/resid_'+str(colours[i])+'.fits',clobber=True)
    #f = open('/data/ljo31/Lens/J1347/coeff'+str(X),'wb')
    #cPickle.dump(coeff,f,2)
    #f.close()
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()

print 'source 1 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n1, '&', '%.2f'%re1, '&', '%.2f'%q1, '&','%.2f'%pa1,  r'\\'


print 'galaxy 1 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n2, '&', '%.2f'%re2, '&', '%.2f'%q2, '&','%.2f'%pa2,  r'\\'

print 'lens 1 ', '&', '%.2f'%x4, '&',  '%.2f'%y4, '&', '%.2f'%eta, '&', '%.2f'%b, '&', '%.2f'%q4, '&','%.2f'%pa4,  r'\\\hline'

print 'shear = ', '%.4f'%shear, 'shear pa = ', '%.2f'%shearpa

