import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee as myEmcee
from emcee import PTSampler

# try basicmodel4 with a load of different position angles for source 1

# plot things
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
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()
    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')


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


guiFile = '/data/ljo31/Lens/J1347/basicmodel2'
#guiFile = '/data/ljo31/Lens/J1347/bm4'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2a'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2c'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2e'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2f'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2g'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2h'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2h2'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2h3'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2h4'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2i'

print guiFile

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



G,L,S,offsets,_ = numpy.load(guiFile)

ntemps,nwalkers,ndim=20,60,26
p0 = np.zeros((ntemps,nwalkers, ndim)) 

### first parameters need to be the offsets
xoffset = offsets[0][3]
yoffset = offsets[1][3]
p0[:,:,0] = xoffset + np.random.randn(ntemps,nwalkers)*0.5 # xoffset
p0[:,:,1] = yoffset + np.random.randn(ntemps,nwalkers)*0.5 # yoffset

#p0[:,:,2] = S['Source 1']['x']['value'] + np.random.randn(ntemps,nwalkers)*2
#p0[:,:,3] = S['Source 1']['y']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,2] = S['Source 1']['q']['value'] + np.random.randn(ntemps,nwalkers)*0.2
p0[:,:,3] = S['Source 1']['pa']['value'] + np.random.randn(ntemps,nwalkers)*70
p0[:,:,4] = S['Source 1']['re']['value'] + np.random.randn(ntemps,nwalkers)
p0[:,:,5] = S['Source 1']['n']['value'] + np.random.randn(ntemps,nwalkers)

p0[:,:,6] = S['Source 2']['x']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,7] = S['Source 2']['y']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,8] = S['Source 2']['q']['value'] + np.random.randn(ntemps,nwalkers)*0.2
p0[:,:,9] = S['Source 2']['pa']['value'] + np.random.randn(ntemps,nwalkers)*70
p0[:,:,10] = S['Source 2']['re']['value'] + np.random.randn(ntemps,nwalkers)*0.5
p0[:,:,11] = S['Source 2']['n']['value'] + np.random.randn(ntemps,nwalkers)*0.5

p0[:,:,12] = G['Galaxy 1']['x']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,13] = G['Galaxy 1']['y']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,14] = G['Galaxy 1']['q']['value'] + np.random.randn(ntemps,nwalkers)*0.2
p0[:,:,15] = G['Galaxy 1']['pa']['value'] + np.random.randn(ntemps,nwalkers)*10
p0[:,:,16] = G['Galaxy 1']['re']['value'] + np.random.randn(ntemps,nwalkers)*1
p0[:,:,17] = G['Galaxy 1']['n']['value'] + np.random.randn(ntemps,nwalkers)*0.5

p0[:,:,18] = L['Lens 1']['x']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,19] = L['Lens 1']['y']['value'] + np.random.randn(ntemps,nwalkers)*2
p0[:,:,20] = L['Lens 1']['q']['value'] + np.random.randn(ntemps,nwalkers)*0.2
p0[:,:,21] = L['Lens 1']['pa']['value'] + np.random.randn(ntemps,nwalkers)*10
p0[:,:,22] = L['Lens 1']['b']['value'] + np.random.randn(ntemps,nwalkers)*0.5
p0[:,:,23] = L['Lens 1']['eta']['value'] + np.random.randn(ntemps,nwalkers)*0.1

p0[:,:,24] = 0.0 + np.random.randn(ntemps,nwalkers)*0.04 # external shear
p0[:,:,25] = 0 + np.random.randn(ntemps,nwalkers)*110 # shear PA


def lnprob(X):
    dx,dy,q1,pa1,re1,n1,x2,y2,q2,pa2,re2,n2,x3,y3,q3,pa3,re3,n3,x4,y4,q4,pa4,b,eta,shear,shearpa = X
    if n1 < 0.1 or n2 < 0.1 or n3 < 0.1 or np.any(np.array([q1,q2,q3,q4,re1,re2,re3,b,eta])<0) or  np.any(np.array([q1,q2,q3,q4])>1):
        return -np.inf
    srcs, gals, lenses = [], [], []
    srcs.append(SBModels.Sersic('Source 1', {'x':x2,'y':y2,'q':q1,'pa':pa1,'re':re1,'n':n1}))
    srcs.append(SBModels.Sersic('Source 2', {'x':x2,'y':y2,'q':q2,'pa':pa2,'re':re2,'n':n2}))
    gals.append(SBModels.Sersic('Galaxy 1', {'x':x3,'y':y3,'q':q3,'pa':pa3,'re':re3,'n':n3}))
    lenses.append(MassModels.PowerLaw('Lens 1', {'x':x4,'y':y4,'q':q4,'pa':pa4,'b':b,'eta':eta}))
    lenses.append(MassModels.ExtShear('shear',{'x':x4,'y':y4,'b':shear, 'pa':shearpa}))
    lp = 0.
    OVRS=4
    for i in range(len(imgs)):
        if i == 0:
            x0,y0 = 0,0
        else:
            x0,y0 = dx,dy
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        lp += lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,verbose=False,psf=psf,csub=1)
    #print lp
    return lp

def logp(X):
    return 0

sampler=PTSampler(ntemps, nwalkers, ndim, lnprob,logp,threads=4)
for p, lnprob, lnlike in sampler.sample(p0, iterations=200):
    pass
sampler.reset()
print 'sampled'
for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike,iterations=200):
    pass
assert sampler.chain.shape == (ntemps, nwalkers, 200, ndim)
print 'fertig?'
'''
outFile = '/data/ljo31/Lens/J1347/test_emcee'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

result = S.result()
lp = result[0]
trace = numpy.array(result[1])
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

## now we need to interpret these resultaeten
logp,coeffs,dic,vals = result
ii = np.where(logp==amax(logp))
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
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))



ims = []
models = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = det['xoffset'][-1], det['yoffset'][-1] # xoffset, yoffset #
        print x0,y0
    im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True) # return loglikelihood
    print im
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
    pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1347/components_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1347/model_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1347/resid_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    f = open('/data/ljo31/Lens/J1347/coeff'+str(X),'wb')
    cPickle.dump(coeff,f,2)
    f.close()
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))


#numpy.save('/data/ljo31/Lens/J1606/trace'+str(Y), trace)
#numpy.save('/data/ljo31/Lens/J1606/logP'+str(Y), logp)

for key in det.keys():
    print key, '%.1f'%det[key][-1]
print 'max lnL is ', max(logp)

print det['xoffset'], det['yoffset']
np.save('/data/ljo31/Lens/J1347/det'+str(X),det)

#print 'x & y & n & re & q & pa \\'
print '&','&', '%.1f'%det['Source 1 n'][-1], '&', '%.1f'%det['Source 1 re'][-1], '&', '%.1f'%det['Source 1 q'][-1], '&', '%.1f'%det['Source 1 pa'][-1], '\\'
print '%.1f'%det['Source 2 x'][-1], '&', '%.1f'%det['Source 2 y'][-1],' &', '%.1f'%det['Source 2 n'][-1], '&', '%.1f'%det['Source 2 re'][-1], '&', '%.1f'%det['Source 2 q'][-1], '&', '%.1f'%det['Source 2 pa'][-1], '\\'
print '%.1f'%det['Galaxy 1 x'][-1], '&', '%.1f'%det['Galaxy 1 y'][-1], '&', '%.1f'%det['Galaxy 1 n'][-1], '&', '%.1f'%det['Galaxy 1 re'][-1], '&', '%.1f'%det['Galaxy 1 q'][-1], '&', '%.1f'%det['Galaxy 1 pa'][-1], '\\'
print '%.1f'%det['Lens 1 x'][-1], '&', '%.1f'%det['Lens 1 y'][-1], '&', '%.1f'%det['Lens 1 eta'][-1], '&', '%.1f'%det['Lens 1 b'][-1], '&', '%.1f'%det['Lens 1 q'][-1], '&', '%.1f'%det['Lens 1 pa'][-1], '\\'



## now we can extract the results and construct the source galaxy in the source plane!
srcs = []
p1,p2 = {},{}
for name in det.keys():
    s = det[name]
    if name[:8] == 'Source 1':
        for key in 'x','y','q','pa','re','n':
            p1[key] = s[-1]
    elif name[:8] == 'Source 2':
        for key in 'x','y','q','pa','re','n':
            p2[key] = s[-1]
               
srcs.append(SBModels.Sersic('Source 1',p1))
srcs.append(SBModels.Sersic('Source 2',p2))
ims = []
tims = np.zeros(imgs.shape)
for i in range(len(srcs)):
    src = srcs[i]
    im = src.pixeval(xc,yc)
    ims.append(im)
    tims +=im

pl.figure()
pl.imshow(tims,origin-'lower',interpolation='nearest')
for i in range(2):
    pl.figure()
    pl.imshow(ims[i],origin='lower',interpolation='nearest')

'''
