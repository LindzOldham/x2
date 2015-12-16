import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
from pylens import lensModel

# the plan here is to infer the model and the Keck PSF all at once, using all three bands of data
# so we require two sets of offsets, one more dramatic than the other, and two sets of coordinates to account for the different sizes and pixel scales of the HST and Keck dataz.

def NotPlicely(image,im,sigma,vmax):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=vmax,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    #pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=vmax,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    #pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25,cmap = 'afmhot',aspect='auto')
    pl.colorbar()
    #pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    #pl.title('signal-to-noise residuals')
    pl.colorbar()
    pl.subplots_adjust(left=0.05,bottom=0.05,top=0.95,right=0.95)
    pl.subplots_adjust(wspace=0.1,hspace=0.1)




# 0.02 arcsec/pixel
# HST: 0.05 arcsec/pixel --> so need to scale and somehow recentre.
# p_k = p_h * 2.5

img1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2_masked.fits')[0].data.copy() # masking out the possiblyasecondsource regions!
psf1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf2= psf2[15:-16,14:-16]
psf2 /= psf2.sum()

img3 = pyfits.open('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_cutout.fits')[0].data.copy()
sig3 = pyfits.open('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_sigma.fits')[0].data.copy()
yc3,xc3 = iT.coords(img3.shape)*0.2
xp,yp = iT.coords((171,171))-85
OVRS = 1

imgs = [img1,img2,img3]
sigs = [sig1,sig2,sig3]
psfs = [psf1,psf2]

PSFs = []
yc,xc = iT.overSample(img1.shape,1.)
yc,xc = yc-15.,xc-15.
for i in range(len(psfs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)


guiFile = '/data/ljo31/Lens/J1605/terminal_iterated_4'

G,L,S,offsets,_ = numpy.load(guiFile)

pars = []
cov = []

### first parameters relate to the PSF
pars.append(pymc.Uniform('xoffset - keck',9,13,value=10.5))
pars.append(pymc.Uniform('yoffset - keck',10,14,value=12))
cov += [5,5] # think about this!
pars.append(pymc.Uniform('sigma 1',0,8,value=2))
cov += [5]
pars.append(pymc.Uniform('q 1',0,1,value=0.7))
cov += [1]
pars.append(pymc.Uniform('pa 1',-180,180,value= 90 )) #-30 )) #-30.0))
cov += [50]
pars.append(pymc.Uniform('amp 1',0,1,value=0.7))
cov += [4]
pars.append(pymc.Uniform('sigma 2',0,150,value= 15 )) #15 )) #15))
cov += [40]
pars.append(pymc.Uniform('q 2',0,1,value=0.9))
cov += [1]
pars.append(pymc.Uniform('pa 2',-180,180,value= 90 )) #-100 )) ##-100))
cov += [100]

'''pars.append(pymc.Uniform('xoffset - keck',9,13,value=10.5))
pars.append(pymc.Uniform('yoffset - keck',10,14,value=12))
cov += [1,1] # think about this!
pars.append(pymc.Uniform('sigma 1',2,8,value=4))
cov += [1]
pars.append(pymc.Uniform('q 1',0,1,value=0.7))
cov += [1]
pars.append(pymc.Uniform('pa 1',-180,180,value= 90 )) #-30 )) #-30.0))
cov += [10]
pars.append(pymc.Uniform('amp 1',0,1,value=0.7))
cov += [0.5]
pars.append(pymc.Uniform('sigma 2',0,50,value= 30 )) #15 )) #15))
cov += [5]
pars.append(pymc.Uniform('q 2',0,1,value=0.9))
cov += [0.5]
pars.append(pymc.Uniform('pa 2',-180,180,value= 90 )) #-100 )) ##-100))
cov += [100]'''


### now the img1 and img2 params
xoffset = offsets[0][3]
yoffset = offsets[1][3]

pars.append(pymc.Uniform('xoffset',-2.,2.,value=offsets[0][3]))
pars.append(pymc.Uniform('yoffset',-2.,2.,value=offsets[1][3]))
cov += [0.4,0.4]

srcs = []
for name in 'Source 1', 'Source 2':
    s = S[name]
    p = {}
    if name == 'Source 1':
        for key in 'x','y','q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
    elif name == 'Source 2':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))

gals = []
for name in 'Galaxy 1', 'Galaxy 2':
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

lenses = []
for name in L.keys():
    s = L[name]
    p = {}
    for key in 'x','y','q','pa','b','eta':
        if s[key]['type']=='constant':
            p[key] = s[key]['value']
        else:
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            if key == 'pa':
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=0))
cov.append(0.01)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=0.))
cov.append(5.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

print pars
print cov

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0,p=pars):
    lp = 0.
    for i in range(3):
        #print i
        image = imgs[i]
        sigma = sigs[i]
        if i == 0:
            psf = PSFs[i]
            x0,y0 = 0,0
            xm,ym = xc,yc
            #print i, x0,y0
        if i == 1:
            psf = PSFs[i]
            x0,y0 = pars[9].value,pars[10].value
            #print i, x0,y0
            xm,ym = xc,yc
        elif i ==2:
            x0,y0 = pars[0].value,pars[1].value
            #print i,x0,y0
            sigma1 = pars[2].value.item()
            q1 = pars[3].value.item()
            pa1 = pars[4].value.item()
            amp1 = pars[5].value.item()
            sigma2 = pars[6].value.item()
            q2 = pars[7].value.item()
            pa2 = pars[8].value.item()
            amp2 = 1.-amp1
            psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':sigma1,'q':q1,'pa':pa1,'amp':10})
            psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':sigma2,'q':q2,'pa':pa2,'amp':10})
            psf1 = psfObj1.pixeval(xp,yp) * amp1 / (np.pi*2.*sigma1**2.)
            psf2 = psfObj2.pixeval(xp,yp) * amp2 / (np.pi*2.*sigma2**2.)
            psf = psf1 + psf2
            psf /= psf.sum()
            psf = convolve.convolve(image,psf)[1]
            xm,ym = xc3,yc3
        lp += lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xm+x0,ym+y0,1,
                                verbose=False,psf=psf,csub=1)
       
    return lp
        
    

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
    S.sample(100*len(pars)**2)

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
sigma = pyfits.open('/data/ljo31/Lens/J1605/J1605_Kp_narrow_med_sigma.fits')[0].data.copy()
x0 = det['xoffset - keck'][-1]
y0 = det['yoffset - keck'][-1]
sig1 = det['sigma 1'][-1]
q1 = det['q 1'][-1]
pa1 = det['pa 1'][-1]
amp1 = det['amp 1'][-1]
sig2 = det['sigma 2'][-1]
q2 = det['q 2'][-1]
pa2 = det['pa 2'][-1]
amp2 = 1.-amp1
psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':sig1,'q':q1,'pa':pa1,'amp':10})
psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':sig2,'q':q2,'pa':pa2,'amp':10})
psf1 = psfObj1.pixeval(xp,yp) * amp1 / (2.*np.pi*sig1**2)
psf2 = psfObj2.pixeval(xp,yp) * amp2 / (2.*np.pi*sig2**2)
psf = psf1 + psf2
psf /= psf.sum()
psf = convolve.convolve(img3,psf)[1]
im = lensModel.lensFit(None,img3,sigma,gals,lenses,srcs,xc3+x0,yc3+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
model = lensModel.lensFit(None,img3,sigma,gals,lenses,srcs,xc3+x0,yc3+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True)

NotPlicely(img3,im,sigma,8)

ims = []
models = []
for i in range(len(PSFs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = det['xoffset'][-1], det['yoffset'][-1] # xoffset, yoffset #
        print x0,y0
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True) # return loglikelihood
    print im
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    ims.append(im)
    models.append(model)

colours = ['F555W', 'F814W']
vmaxes = [1.2,6.4]
for i in range(len(PSFs)):
    image = imgs[i]
    im = ims[i]
    model = models[i]
    sigma = sigs[i]
    NotPlicely(image,im,sigma,vmaxes[i])
    pl.suptitle(str(colours[i]))


for key in det.keys():
    print key, det[key][-1]
print 'x & y & n & re & q & pa \\'
print '%.1f'%det['Source 1 x'][-1], '&', '%.1f'%det['Source 1 y'][-1], '&', '%.1f'%det['Source 1 n'][-1], '&', '%.1f'%det['Source 1 re'][-1], '&', '%.1f'%det['Source 1 q'][-1], '&', '%.1f'%det['Source 1 pa'][-1], '\\'
print '& & &', '%.1f'%det['Source 2 n'][-1], '&', '%.1f'%det['Source 2 re'][-1], '&', '%.1f'%det['Source 2 q'][-1], '&', '%.1f'%det['Source 2 pa'][-1], '\\'
print '%.1f'%det['Galaxy 1 x'][-1], '&', '%.1f'%det['Galaxy 1 y'][-1], '&', '%.1f'%det['Galaxy 1 n'][-1], '&', '%.1f'%det['Galaxy 1 re'][-1], '&', '%.1f'%det['Galaxy 1 q'][-1], '&', '%.1f'%det['Galaxy 1 pa'][-1], '\\'
print '& & &', '%.1f'%det['Galaxy 2 n'][-1], '&', '%.1f'%det['Galaxy 2 re'][-1], '&', '%.1f'%det['Galaxy 2 q'][-1], '&', '%.1f'%det['Galaxy 2 pa'][-1], '\\'
print '%.1f'%det['Lens 1 x'][-1], '&', '%.1f'%det['Lens 1 y'][-1], '&', '%.1f'%det['Lens 1 eta'][-1], '&', '%.1f'%det['Lens 1 b'][-1], '&', '%.1f'%det['Lens 1 q'][-1], '&', '%.1f'%det['Lens 1 pa'][-1], '\\'
print 'KECK - x & y & sigma 1 & sigma 2 & pa 1 & pa 2 & q 1 & q 2 & amp 1 \\'
print '%.1f'%det['xoffset - keck'][-1], '&', '%.1f'%det['yoffset - keck'][-1], '&', '%.1f'%det['sigma 1'][-1], '&', '%.1f'%det['sigma 2'][-1], '&', '%.1f'%det['pa 1'][-1], '&', '%.1f'%det['pa 2'][-1], '&', '%.1f'%det['q 1'][-1], '&', '%.1f'%det['q 2'][-1], '&', '%.1f'%det['amp 1'][-1], '\\'


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

