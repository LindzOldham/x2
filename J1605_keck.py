import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
from pylens import lensModel


# 0.02 arcsec/pixel
# HST: 0.05 arcsec/pixel --> so need to scale and somehow recentre.
# p_k = p_h * 2.5

def MakeCuts():
    image = pyfits.open('/data/mauger/EELs/SDSSJ1605+3811/J1605_Kp_narrow_med.fits')[0].data.copy()[550:750,600:850]
    header = pyfits.open('/data/mauger/EELs/SDSSJ1605+3811/J1605_Kp_narrow_med.fits')[0].header
    pyfits.writeto('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_cutout.fits',image,header,clobber=True)

def MakeMaps():
    image = pyfits.open('/data/mauger/EELs/SDSSJ1605+3811/J1605_Kp_narrow_med.fits')[0].data.copy()
    header = pyfits.open('/data/mauger/EELs/SDSSJ1605+3811/J1605_Kp_narrow_med.fits')[0].header
    cut1 = image[680:725,525:575]
    cut2 = image[700:750,850:925]
    cut3 = image[525:575,865:950]
    var1,var2,var3 = np.var(cut1),np.var(cut2),np.var(cut3)
    poisson = np.mean((var1,var2,var3))
    sigma = poisson**0.5
    im = pyfits.open('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_cutout.fits')[0].data.copy()
    smooth = ndimage.gaussian_filter(im,0.7)
    noisemap = np.where((smooth>0.7*sigma)&(im>0),im/120.+poisson, poisson)**0.5 # for now - nut find out the actual exposure time from Matt...
    pyfits.writeto('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_sigma.fits',noisemap,header,clobber=True)
    pl.figure()
    pl.imshow(noisemap)
    pl.colorbar()

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25)
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5)
    pl.title('signal-to-noise residuals')
    pl.colorbar()
    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')

image = pyfits.open('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_cutout.fits')[0].data.copy()
sigma = pyfits.open('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_sigma.fits')[0].data.copy()
yc,xc = iT.coords(image.shape)

# Model the PSF as a Gaussian to start with. We'll do this over a grid of sigmas, and then maybe also ellitpicity and position anlge (will get kompliziert!!)
xp,yp = iT.coords((150,150))-75
#print yp.shape,xp.shape
#sig=1 #4.
#psfObj = SBObjects.Gauss('psf',{'x':0,'y':0,'sigma':sig,'q':1,'pa':0,'amp':1})
#psf = psfObj.pixeval(xp,yp)
#psf /= psf.sum()
#psf = convolve.convolve(image,psf)[1]

OVRS = 1

# we should get these from the iterated terminal version.
det = np.load('/data/ljo31/Lens/J1605/det8.npy')[()]
g1,g2,l1,s1,s2,sh = {},{},{},{},{},{}

#srcs = []
#gals = []
#lenses = []
coeff=[]
for name in det.keys():
    s = det[name]
    coeff.append(s[-1])
    if name[:8] == 'Source 1':
        s1[name[9:]] = s[-1]
    elif name[:8] == 'Source 2':
        s2[name[9:]] = s[-1]
    elif name[:6] == 'Lens 1':
        l1[name[7:]] = s[-1]
    elif name[:8] == 'Galaxy 1':
        g1[name[9:]] = s[-1]
    elif name[:8] == 'Galaxy 2':
        g2[name[9:]] = s[-1]
    elif name[:8] == 'extShear':
        if len(name)<9:
            sh['b'] = s[-1]
        elif name == 'extShear PA':
            sh['pa'] = s[-1]
       
g1['x'],g2['x'] = g1['x']+67,g2['x']+67
g1['y'],g2['y'] = g1['y']+62.5,g2['y']+62.5
s1['x'] = g1['x'] + 5*(s1['x']-(g1['x']-67))
s2['x'] = g1['x'] + 5*(s2['x']-(g1['x']-67))
s1['y'] = g1['y'] + 5*(s1['y']-(g1['y']-62.5))
s2['y'] = g1['y'] + 5*(s2['y']-(g1['y']-62.5))
l1['x'] = g1['x'] + 5*(l1['x']-(g1['x']-67))
l1['y'] = g1['y'] + 5*(l1['y']-(g1['y']-62.5))
s1['re'],s2['re'],g1['re'],g2['re'],l1['b'],sh['b'] = s1['re']*5,s2['re']*5,g1['re']*5,g2['re']*5,l1['b']*5,sh['b']*5 

#srcs.append(SBModels.Sersic('Source 1',s1))
#srcs.append(SBModels.Sersic('Source 2',s2))
#gals.append(SBModels.Sersic('Galaxy 1',g1))
#gals.append(SBModels.Sersic('Galaxy 2',g2))
lenses.append(MassModels.PowerLaw('Lens 1',l1))

sh['x'] = lenses[0].pars['x']
sh['y'] = lenses[0].pars['y']
#lenses.append(MassModels.ExtShear('shear',sh))

#lp = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc,yc,1,verbose=False,psf=psf,csub=1) # eventually, construct a grid of psfs and iterate over this!
#im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True)
#pl.figure()
#pl.subplot(211)
#pl.imshow(im,origin='lower',interpolation='nearest')
#pl.colorbar()
#pl.subplot(212)
#pl.imshow(image,origin='lower',interpolation='nearest',vmin=0)
#pl.colorbar()


## try some way of fitting!
#pars = [xoffset,yoffset,sig]
pars = []
cov = []
pars.append(pymc.Uniform('xoffset',57.,77.,value=67))
pars.append(pymc.Uniform('yoffset',52.5,72.5,value=62.5))
cov += [20,20] # think about this!
pars.append(pymc.Uniform('sigma',0,10,value=4))
cov += [5]

@pymc.deterministic
def logP(value=0,p=pars):
    x0 = pars[0].value
    y0 = pars[1].value
    sig = pars[2].value
    psfObj = SBObjects.Gauss('psf',{'x':0,'y':0,'sigma':sig,'q':1,'pa':0,'amp':1})
    psf = psfObj.pixeval(xp,yp)
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    
    g1a,g2a,s1a,s2a,l1a,sha = g1.copy(),g2.copy(),s1.copy(),s2.copy(),l1.copy(),sh.copy()
    g1a['x'],g2a['x'] = g1a['x']+x0,g2a['x']+x0
    g1a['y'],g2a['y'] = g1a['y']+y0,g2a['y']+y0
    s1a['x'] = g1a['x'] + 5*(s1a['x']-(g1a['x']-x0))
    s2a['x'] = g1a['x'] + 5*(s2a['x']-(g1a['x']-x0))
    s1a['y'] = g1a['y'] + 5*(s1a['y']-(g1a['y']-y0))
    s2a['y'] = g1a['y'] + 5*(s2a['y']-(g1a['y']-y0))
    l1a['x'] = g1a['x'] + 5*(l1a['x']-(g1a['x']-x0))
    l1a['y'] = g1a['y'] + 5*(l1a['y']-(g1a['y']-y0))
    s1a['re'],s2a['re'],g1a['re'],g2a['re'],l1a['b'],sha['b'] = s1a['re']*5,s2a['re']*5,g1a['re']*5,g2a['re']*5,l1a['b']*5,sha['b']*5 
    srcs = []
    gals = []
    lenses = []
    srcs.append(SBModels.Sersic('Source 1',s1a))
    srcs.append(SBModels.Sersic('Source 2',s2a))
    gals.append(SBModels.Sersic('Galaxy 1',g1a))
    gals.append(SBModels.Sersic('Galaxy 2',g2a))
    lenses.append(MassModels.PowerLaw('Lens 1',l1a))

    sha['x'] = lenses[0].pars['x']
    sha['y'] = lenses[0].pars['y']
    lenses.append(MassModels.ExtShear('shear',sha))
    return lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,1,
                                verbose=False,psf=psf,csub=1)

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)

# use lensFit to calculate the likelihood at each point in the chain
for i in range(1):
    S = AMAOpt(pars,[likelihood],[logP],cov=optCov/4.)
    S.set_minprop(len(pars)*2)
    S.sample(10*len(pars)**2)

logp,trace,det = S.result() # log likelihoods; chain (steps * params); det['extShear PA'] = chain in this variable
coeff = []
for i in range(len(pars)):
    coeff.append(trace[-1,i])

coeff = numpy.asarray(coeff)
pars = coeff
o = 'npars = ['
for i in range(pars.size):
    o += '%f,'%(pars)[i]
o = o[:-1]+"]"

keylist = []
dkeylist = []
chainlist = []
for key in det.keys():
    keylist.append(key)
    dkeylist.append(det[key][-1])
    chainlist.append(det[key])

plot = True
if plot:
    for i in range(len(keylist)):
        pl.figure()
        pl.plot(chainlist[i])
        pl.title(str(keylist[i]))

# compare best model with data!
x0 = det['xoffset'][-1]
y0 = det['yoffset'][-1]
sig = det['sigma'][-1]



'''



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

colours = ['F555W', 'F814W']
for i in range(len(imgs)):
    image = imgs[i]
    im = ims[i]
    model = models[i]
    sigma = sigs[i]
    pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1605/components_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1605/model_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1605/resid_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    f = open('/data/ljo31/Lens/J1605/coeff'+str(X),'wb')
    cPickle.dump(coeff,f,2)
    f.close()
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))


### OUTPUT THE THINGS IN LATEX-FRIENDLY FORM!
#print '%.1f'%det['Source 1 x'][-1], '&', '%.1f'%det['Source 1 y'][-1], '&', '%.1f'%det['Source 1 n'][-1], '&', '%.1f'%det['Source 1 re'][-1], '&', '%.1f'%det['Source 1 q'][-1], '&', '%.1f'%det['Source 1 pa'][-1], '\\'
#print '%.1f'%det['Source 2 n'][-1], '&', '%.1f'%det['Source 2 re'][-1], '&', '%.1f'%det['Source 2 q'][-1], '&', '%.1f'%det['Source 2 pa'][-1], '\\'

#numpy.save('/data/ljo31/Lens/J1606/trace'+str(Y), trace)
#numpy.save('/data/ljo31/Lens/J1606/logP'+str(Y), logp)

for key in det.keys():
    print key, '%.1f'%det[key][-1]
print 'max lnL is ', max(logp)

#pl.figure()
#pl.imshow((im-image)/sigma)
#pl.colorbar()
print det['xoffset'], det['yoffset']
#print xoffset, yoffset

'''
