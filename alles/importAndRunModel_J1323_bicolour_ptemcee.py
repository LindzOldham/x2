import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from matplotlib.colors import LogNorm

'''
X = 0 - basicmodel1
X = 1 - basicmodel3
X = 2 - basicmodel6 
X = 3 - basicmodel8. Two sources, and with the lens having ellipticity. Kind of arbitrary though: just picked a position anglew by rotating and having a look.
X = 4 - basicmodel8. Oversampling = 2.
X = 5 - x3. That is, basicmodel8 with this code already run on it once, then put back into the gui.
X = 6 - x3_iterated. This is x3, but put through the gui for a while.
X = 7 - basicmodel8d. Meant to do this earlier. BAD
X = 8 - x3_iterated, with re_smallsource = 0.8 
X = 9 - x3_iterated, with psf3 instead of psf2 to make sure the psf isn't having a big effect on the fit!!!
X = 10 - basicmodel9uniform. This is with three galaxy components! It's a spiral galaxym so perhaps this is kind of justified. Will it help? *Changed from 60 to 80 walkers to explore the larger parameter space.
X = 11 - basicmodel10uniform - three galaxies, but MORE IMPORTANTLY F814w_psf3! This seems to work a lot better?
X = 12 - basicmodel11. This is x3_iterated2, only with psf3 instead of psf2 for the F814W image!
X = 13 - basicmodel12. More iterations!
X = 14 - basicmodel13. More changes!
X = 15 - basicmodel12b -has the lower bound of prior on re1 set to zero, and oversampling - XX
X = 16 - basicmodel12c. This is X = 13 put back into the gui. (Corrected offsets by hand) - XX
X = 17 - a grand inference with basicmodel12, but loads of threads and temperatures, oversampling and better proior on re1. - XX
X = 18 - all of the above were using the wrong model. Now using X = 14, so basicmodel13! (1) A grand inference. bm13_iterated
X = 19 - a grand inference from basicmodel13. This is ready to run as soon as there are some free cores!!! (Need nine)
X = 20 - as 18, only emcee not ptemcee, just to check it's sampling ok!
X = 21 - masking regions using mask2_F814W.fits
'''


X = 21


print X

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',norm=LogNorm()) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',norm=LogNorm()) #,vmin=vmin,vmax=vmax)
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

guiFile = '/data/ljo31/Lens/J1323/basicmodel1'
guiFile = '/data/ljo31/Lens/J1323/basicmodel3'
guiFile = '/data/ljo31/Lens/J1323/basicmodel6'
guiFile = '/data/ljo31/Lens/J1323/basicmodel8'
guiFile = '/data/ljo31/Lens/J1323/x3'
guiFile = '/data/ljo31/Lens/J1323/x3_iterated'
#guiFile = '/data/ljo31/Lens/J1323/basicmodel8d'
guiFile = '/data/ljo31/Lens/J1323/basicmodel9uniform'
guiFile = '/data/ljo31/Lens/J1323/basicmodel10uniform'
guiFile = '/data/ljo31/Lens/J1323/basicmodel11'
guiFile = '/data/ljo31/Lens/J1323/basicmodel12'
guiFile = '/data/ljo31/Lens/J1323/basicmodel13'
guiFile = '/data/ljo31/Lens/J1323/basicmodel12b'
guiFile = '/data/ljo31/Lens/J1323/basicmodel12c'
#guiFile = '/data/ljo31/Lens/J1323/basicmodel12'
guiFile = '/data/ljo31/Lens/J1323/bm13_iterated'
#guiFile = '/data/ljo31/Lens/J1323/basicmodel13'


print guiFile

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 3 # can return to oversampling if need be!
yc,xc = iT.overSample(img1.shape,OVRS)
yc,xc = yc,xc

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)



G,L,S,offsets,_ = numpy.load(guiFile)

pars = []
cov = []
### first parameters need to be the offsets
xoffset = offsets[0][3]
yoffset = offsets[1][3]
pars.append(pymc.Uniform('xoffset',-5.,5.,value=offsets[0][3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=offsets[1][3]))
cov += [0.4,0.4]


srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        for key in 'x','y','q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                if key == 're':
                    pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                else:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                   cov.append(s[key]['sdev']*100) 
                else:
                    cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                   cov.append(s[key]['sdev']*100) 
                else:
                    cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))
'''
srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        for key in 'x','y','q','re','n','pa':
            if key == 'pa':
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=90))
                p[key] = pars[-1]
                cov.append(s[key]['sdev']*10)
            elif key == 'q':
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=0.7))
                p[key] = pars[-1]
                cov.append(s[key]['sdev']*2)
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            if key == 'pa':
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=90))
                p[key] = pars[-1]
                cov.append(s[key]['sdev']*10)
            elif key == 'q':
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=0.7))
                p[key] = pars[-1]
                cov.append(s[key]['sdev']*2)
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))

for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        for key in 'x','y','q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            if key == 're':
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=0.8))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                   cov.append(s[key]['sdev']*100) 
                else:
                    cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                   cov.append(s[key]['sdev']*100) 
                else:
                    cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))


'''

gals = []
for name in G.keys():
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
    '''elif name == 'Galaxy 3':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = gals[0].pars[key]'''
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
                cov.append(s[key]['sdev']*100)
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                cov.append(s[key]['sdev'])
            p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=0.01))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=12))
cov.append(100.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

print len(pars), len(cov)
for p in pars:
    print p

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    for i in range(len(imgs)):
        if i == 0:
            x0,y0 = 0,0
        else:
            x0 = pars[0].value 
            y0 = pars[1].value 
            #print x0,y0
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        lp += lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,verbose=False,psf=psf,csub=1)
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


S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=6,nwalkers=60,ntemps=3)
S.sample(800)
#S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=60)
#S.sample(100)

outFile = '/data/ljo31/Lens/J1323/emcee'+str(X)
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
    pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1323/components_uniform'+str(colours[i])+'.fits',clobber=True)
    pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1323/model_uniform'+str(colours[i])+'.fits',clobber=True)
    pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1323/resid_uniform'+str(colours[i])+'.fits',clobber=True)
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()


