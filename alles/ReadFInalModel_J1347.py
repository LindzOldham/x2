import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
import pylab as pl
import numpy as np

def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
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
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-3,vmax=3,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals 2')
    pl.colorbar()

def SotPleparately(image,im,sigma,col):
    ext = [0,image.shape[0],0,image.shape[1]]
    pl.figure()
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data - '+str(col))
    pl.figure()
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model - '+str(col))
    pl.figure()
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot')
    pl.title('signal-to-noise residuals - '+str(col))
    pl.colorbar()



result = np.load('/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties')
result = np.load('/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_TWO')

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
OVRS = 4
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

print lp
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
    SotPleparately(image,im,sigma,colours[i])
    pl.suptitle(str(colours[i]))
    pl.show()

print 'source 1 ', '&', '%.2f'%x1, '&', '%.2f'%y1,'&','%.2f'%n1, '&','%.2f'%re1,'&','%.2f'%q1,'&','%.2f'%pa1, '\\'
print 'source 2', '&', '%.2f'%x1, '&', '%.2f'%y1,'&','%.2f'%n2, '&','%.2f'%re2,'&','%.2f'%q2,'&','%.2f'%pa2, '\\'
print 'galaxy 1', '&', '%.2f'%x3, '&', '%.2f'%y3,'&','%.2f'%n3, '&','%.2f'%re3,'&','%.2f'%q3,'&','%.2f'%pa3, '\\'
print 'lens 1', '&', '%.2f'%x4, '&', '%.2f'%y4,'&','%.2f'%eta, '&','%.2f'%b,'&','%.2f'%q4,'&','%.2f'%pa4, '\\'
print dx,dy,shear,shearpa

tracer = np.zeros((60*2500,26))
for i in range(26):
    arr = trace[500:,:,i]
    arr = np.ravel(arr)
    tracer[:,i] = arr
    

#import triangle
#fig = triangle.corner(tracer,plot_datapoints = False, plot_contours = True)

upperlower = map(lambda v: (v[1]-v[0],v[1],v[2]-v[1]),zip(*np.percentile(tracer,[16,50,84],axis=0)))
upperlower = np.array(upperlower)

dxa,dya,x1a,y1a,q1a,pa1a,re1a,n1a,q2a,pa2a,re2a,n2a,x3a,y3a,q3a,pa3a,re3a,n3a,x4a,y4a,q4a,pa4a,ba,etaa,sheara,shearpaa = upperlower[:,0]
dxb,dyb,x1b,y1b,q1b,pa1b,re1b,n1b,q2b,pa2b,re2b,n2b,x3b,y3b,q3b,pa3b,re3b,n3b,x4b,y4b,q4b,pa4b,bb,etab,shearb,shearpab = upperlower[:,2]



print 'source 1 ', '& $', '%.2f'%x1, '_{-', '%.2f'%x1a, '}^{+', '%.2f'%x1b, '}$ & $', '%.2f'%y1,'_{-', '%.2f'%y1a, '}^{+', '%.2f'%y1b, '}$ & $','%.2f'%n1, '_{-', '%.2f'%n1a, '}^{+','%.2f'%n1b, '}$ & $','%.2f'%re1,'_{-', '%.2f'%re1a, '}^{+', '%.2f'%re1b, '}$ & $','%.2f'%q1,'_{-', '%.2f'%q1a, '}^{+', '%.2f'%q1b, '}$  & $','%.2f'%pa1, '_{-', '%.2f'%pa1a, '}^{+', '%.2f'%pa1b, '}$', r'\\'

print 'source 2 ', '& $', '%.2f'%x1, '_{-', '%.2f'%x1a, '}^{+', '%.2f'%x1b, '}$ & $', '%.2f'%y1,'_{-', '%.2f'%y1a, '}^{+', '%.2f'%y1b, '}$ & $','%.2f'%n2, '_{-', '%.2f'%n2a, '}^{+','%.2f'%n2b, '}$ & $','%.2f'%re2,'_{-', '%.2f'%re2a, '}^{+', '%.2f'%re2b, '}$ & $','%.2f'%q2,'_{-', '%.2f'%q2a, '}^{+', '%.2f'%q2b, '}$ & $','%.2f'%pa2, '_{-', '%.2f'%pa2a, '}^{+', '%.2f'%pa2b, '}$', r'\\\hline'

print 'galaxy 1 ', '& $', '%.2f'%x3, '_{-', '%.2f'%x3a, '}^{+', '%.2f'%x3b, '}$ & $', '%.2f'%y3,'_{-', '%.2f'%y3a, '}^{+', '%.2f'%y3b, '}$ & $','%.2f'%n3, '_{-', '%.2f'%n3a, '}^{+','%.2f'%n3b, '}$ & $','%.2f'%re3,'_{-', '%.2f'%re3a, '}^{+', '%.2f'%re3b, '}$ & $','%.2f'%q3,'_{-', '%.2f'%q3a, '}^{+', '%.2f'%q3b, '}$ & $','%.2f'%pa3, '_{-', '%.2f'%pa3a, '}^{+', '%.2f'%pa3b, '}$', r'\\\hline'

print 'lens 1 ', '& $', '%.2f'%x4, '_{-', '%.2f'%x4a, '}^{+', '%.2f'%x4b, '}$ & $', '%.2f'%y4,'_{-', '%.2f'%y4a, '}^{+', '%.2f'%y4b, '}$ & $','%.2f'%eta, '_{-', '%.2f'%etaa, '}^{+','%.2f'%etab, '}$ & $','%.3f'%b,'_{-', '%.3f'%ba, '}^{+', '%.3f'%bb, '}$ & $','%.2f'%q4,'_{-', '%.2f'%q4a, '}^{+', '%.2f'%q4b, '}$ & $','%.2f'%pa4, '_{-', '%.2f'%pa4a, '}^{+', '%.2f'%pa4b, '}$' , r'\\\hline'

