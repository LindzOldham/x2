import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
#import myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline


''' This code now also calculates the source position relative to the lens rather than relative to the origin. This means that when the lens moves, the source moves with it! I have tested this in so far as it seems to produce the same results on the final inference as before. Should maybe test it on an earlier model incarnation though.'''

'''
X = 0 - model8. A first try, parallel-tempered. Chaining source position to lens position and fitting for its offset. Not subtracting galaxy or masking yet though.
X = 1 - model8 long. Leave to run overnight! Not masking. Let's make a generous mask - this will help a lot. A generous and somewhat useless mask has now been made -- ready for next time!
X = 2 - model11, with a smaller source and a larger covariance in re(source). STarted using mask, and was accidentally masking all the good stuff instead of throwing the bad stuff away!!!
X = 3 - unchained and went crazy
X = 4 - model11 short
X = 5 - model10 short, trying to diagnose the problem
X = 6 - startingf walkers from the end of X = 1 and continuing for 8000 steps!
X = 11 - starting walkers from the end of X = 5 and going for 30000 steps on 24 cores!!!
X = 18 - model16_uniform; emcee12; psf3s!!!
'''
X = '30a'
print X

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

'''
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

'''

img1 = py.open('/data/ljo31/Lens/J1125/F606W_sci_cutout.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1125/F606W_noisemap_edited.fits')[0].data.copy()
psf1 = py.open('/data/ljo31/Lens/J1125/F606W_psf3_filledin.fits')[0].data.copy()
psf1 = psf1[5:-7,5:-6]
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1125/F814W_sci_cutout.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1125/F814W_noisemap_edited.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1125/F814W_psf3_filledin.fits')[0].data.copy()
psf2 = psf2[5:-8,5:-6]
psf2 = psf2/np.sum(psf2)


guiFile = '/data/ljo31/Lens/J1125/model8'
guiFile = '/data/ljo31/Lens/J1125/model10'
#guiFile = '/data/ljo31/Lens/J1125/model0'
#guiFile = '/data/ljo31/Lens/J1125/model11'
guiFile = '/data/ljo31/Lens/J1125/model16_uniform'
guiFile = '/data/ljo31/Lens/J1125/model33f_both'

print guiFile

result = np.load('/data/ljo31/Lens/J1125/emcee_splitsource30a0')
lp= result[0]
a1,a2,a3 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]
dic = result[2]
trace2 = np.zeros((trace.shape[0],trace.shape[1],trace.shape[2],trace.shape[3]+2))
trace2[:,:,:,:30] = trace
print trace.shape

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 3
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
mask = py.open('/data/ljo31/Lens/J1125/mask814.fits')[0].data.copy()
#mask = np.ones(img1.shape)
tck = RectBivariateSpline(xo[0],yo[:,0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2.T
mask2 = mask2==1
mask = mask==1

## also try masking this bad region:
maskX = py.open('/data/ljo31/Lens/J1125/mask_badpix.fits')[0].data.copy() + py.open('/data/ljo31/Lens/J1125/mask_badpix2.fits')[0].data.copy() # masking two areas now
tck = RectBivariateSpline(xo[0],yo[:,0],maskX)
mask2X = tck.ev(xc,yc)
mask2X[mask2X<0.5] = 0
mask2X[mask2X>0.5] = 1
mask2X = mask2X.T

mask2 = ((mask2==1) & (mask2X==0))
mask = ((mask==1) & (maskX==0))

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

G,L,S,offsets,shear = numpy.load(guiFile)

pars = []
cov = []
### first parameters need to be the offsets
xoffset =  dic['xoffset'][a1,a2,a3]
yoffset = dic['yoffset'][a1,a2,a3]
pars.append(pymc.Uniform('xoffset',-5.,5.,value=xoffset))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=yoffset))
cov += [0.4,0.4]

gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            if key == 'pa':
                if val >0:
                    pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                elif val<0:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,0,value=val))
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                val = dic[name+' '+key][a1,a2,a3]
                if key == 're':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                elif key == 'pa':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val+90))
                else:
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
        lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
        val = dic[name+' '+key][a1,a2,a3]
        pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
        cov.append(s[key]['sdev'])
        p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value= dic['extShear'][a1,a2,a3]))
cov.append(0.1)
p['b'] = pars[-1]
if dic['extShear PA'][a1,a2,a3] > 0:
    pars.append(pymc.Uniform('extShear PA',0.,180.,value=dic['extShear PA'][a1,a2,a3]))
elif dic['extShear PA'][a1,a2,a3] < 0:
    pars.append(pymc.Uniform('extShear PA',-180.,0,value=dic['extShear PA'][a1,a2,a3]))
cov.append(100.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        for key in 'q','re','n','pa':
           lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
           val = dic[name+' '+key][a1,a2,a3]
           if key == 're':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
           elif key == 'n':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.3,hi,value=val))
           else:
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']*10) 
           elif key == 're':
               cov.append(s[key]['sdev']*1) 
           else:
               cov.append(s[key]['sdev'])
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi = lo - lenses[0].pars[key].value.item(), hi - lenses[0].pars[key].value.item()
            print lo, hi
            #val = val - lenses[0].pars[key].value.item()
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi+2,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
           lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
           val = dic[name+' '+key][a1,a2,a3]
           if key == 're':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
           elif key == 'n':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.3,hi,value=val))
           else:
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']*10) 
           elif key == 're':
               cov.append(s[key]['sdev']*1) 
           else:
               cov.append(s[key]['sdev'])
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi = lo - lenses[0].pars[key].value.item(), hi - lenses[0].pars[key].value.item()
            print lo, hi
            #val = val - lenses[0].pars[key].value.item()
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi+2,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            cov.append(s[key]['sdev'])
    srcs.append(SBModels.Sersic(name,p))

print len(pars), len(cov), trace.shape
for p in pars:
    print p, p.value.item()

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    models = []
    for i in range(len(imgs)):
        if i == 0:
            dx,dy = 0,0
        else:
            dx = pars[0].value 
            dy = pars[1].value 
        xp,yp = xc+dx,yc+dy
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
        n = 0
        model = np.empty(((len(gals) + len(srcs)),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xc*0.
            tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
            tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        for lens in lenses:
            lens.setPars()
        x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
        for src in srcs:
            src.setPars()
            tmp = xc*0.
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=1)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        rhs = (imin/sigin) # data
        op = (model/sigin).T # model matrix
        fit, chi = optimize.nnls(op,rhs)
        model = (model.T*fit).sum(1)
        resid = (model-imin)/sigin
        lp += -0.5*(resid**2.).sum()
        models.append(model)
    return lp #,models
 
  
@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp #[0]

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)


jj=1
for jj in range(1,4):
    print jj
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=10,nwalkers=70,ntemps=4,initialPars=trace[a1])
    S.sample(5000)
    #S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nwalkers=100,nthreads=6) # should have 100 walkers.
    #S.sample(1000)

    outFile = '/data/ljo31/Lens/J1125/emcee_splitsource'+str(X)+str(jj)
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()

    result = S.result()
    lp = result[0]

    trace = numpy.array(result[1])
    a1,a2,a3 = numpy.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,a3,i]
        print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)
    jj+=1


## now we need to interpret these resultaeten
logp,coeffs,dic,vals = result
ii = np.where(logp==np.amax(logp))
coeff = coeffs[ii][0]

colours = ['F555W', 'F814W']
#mods = S.blobs
models = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
    if i == 0:
        dx,dy = 0,0
    else:
        dx = pars[0].value 
        dy = pars[1].value 
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(X)+' '+str(colours[i]))
    pl.show()

'''
S.sample(3500)
#S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nwalkers=100,nthreads=6) # should have 100 walkers.
#S.sample(1000)

outFile = '/data/ljo31/Lens/J1125/emcee'+str(X)
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

colours = ['F555W', 'F814W']
#mods = S.blobs
models = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
    if i == 0:
        dx,dy = 0,0
    else:
        dx = pars[0].value 
        dy = pars[1].value 
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
'''
