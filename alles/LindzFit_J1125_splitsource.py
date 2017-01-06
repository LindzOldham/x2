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
X = 5 - model10 short
X = 6 - trying to reproduce 0
X = 7 - model14, 12000 steps and 6 temps. Two galaxies now.
X = 8 - model11, 12000 steps and 6 temps. One galaxy, but mabes a better model than model8! RUnning for 8000, then 4000 steps!
X = 9 - model_emcee1_uniforms. Running in steps on 1000, then saving. --- THIS WILL OVERWRITE THE OTHER EIGHT!!! OH NO. Convert this to 9 each time it finishes...!
X = 12 -model16_uniform. Started again in the gui, learning from experience. Now running this for 30000 steps and on hopefully lots of cores! 
X = 14 - model21_torun. This now has two sources in an attempt to deal with the weird ringing thing.
X = 15 - model22 NOT DONE
X = 16 - model23 NOT DONE
X = 17 - model24 NOT DONE
X = 16a - continuing 14 as it was still increasing!
X = 18 - model16_uniform; emcee12; psf3s!!!
X = 19 - model30 - psf3!
X = 20 - model33
X = 21 - model34, pte sampling more widely, multiplying covariances by 10 just to explore a wider region of parameter space. This didn't work AT ALL and the solutions found were al REALLY BAD!
X = 22 - model33c_both - from modelling the bands singly to start with!
X = 23 - as above, only long (30,000 steps instead of 5000!)
X = 24 - model33d_both. OVRS = 2, 7500 steps.
X = 25 - as above, but with OVRS = 1. To see if it makes any difference!
X = 26 - end of X = 22, with OVRS = 2 to see if it does anything!
X = 27 - model33e_both. This is X = 22 put back into the gui and iterated a bit. OVRS = 1
X = 23a - repeating 23 because I don't like its solution. Why are these things all so degenerate?? It's totally different from X =22, its shorter cousin, and appears to fit better but has stupid Sersic parameters. Imposed slightly harsher priors on re's: re(gal) < 60, re(source) > 1. RUN NEXT !!!
X = 28 - model33g_both. This is model33e_both with the single galaxy subtracted and replaced with two galaxies. RUNNING!
X = 29 - model33f_both. Further iterations of model33g_both. DO NEXT
X = 26a - a longlonglong version of X=22/26. Oh dear!
X = 15a - continuing 15. 20,000 steps this time, and note that OVRS=2 is important!
X = opt_15a - continuing 15 with the optimizer rather than pyemcee. 500*900 steps
X = opt_15a_short - as above only shorter. 100*900 steps. (900 steps seems to take a couple of minutes so I think this should be fine.)
X - opt_15a_short2 - using covOpt/4 instead of /2.
X - opt_15a_short3 - using covOpt/1 instead of /2.
X = opt_26b - continuing 26a, but with the optimizer.
X = opt_26b_short - 100*900 steps instead of 500*900
X = opt_15a_long - let's see what happens if we leave it running for like a day. 1000*900 - see if it finishes and what happens!!!
X = 29a - exended version of 29!!!
X = 29b - OVRS = 4 off the end of 29a. Source 1 is basically a point source, so we need to oversample quite a bit here. 
X = 30 - repeating 29 to see if it goes to the same place or if there is perhaps some degeneracy? ovrs = 2.
X = 31 - off the back of 29b, but with galaxy 2 rotated by 90 degrees because q = 0.99 at the moment. This might not work at all! ovrs = 3; set up checkpointing in lind\fit_readboth!
X = 32 - splitsource of model33f_both.
X = 33 - gui_29b_D - put 29b back into the terminal and imposed MASSIVE improvements (I think, anyway!)
X = 34 - gui_29b_C1
'''
X = 36
print X

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
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

#guiFile = '/data/ljo31/Lens/J1125/model8'
#guiFile = '/data/ljo31/Lens/J1125/model10'
#guiFile = '/data/ljo31/Lens/J1125/model0'
#guiFile = '/data/ljo31/Lens/J1125/model11'
#guiFile = '/data/ljo31/Lens/J1125/model14'
#guiFile = '/data/ljo31/Lens/J1125/model11'
#guiFile = '/data/ljo31/Lens/J1125/model_emcee1_uniforms' # run this next!!!
#guiFile = '/data/ljo31/Lens/J1125/model21_torun' # run this next!!!
#guiFile = '/data/ljo31/Lens/J1125/model22' # run this next!!!
#guiFile = '/data/ljo31/Lens/J1125/model23' # run this next!!!
#guiFile = '/data/ljo31/Lens/J1125/model22'
#guiFile = '/data/ljo31/Lens/J1125/model31'
#guiFile = '/data/ljo31/Lens/J1125/model33'
#guiFile = '/data/ljo31/Lens/J1125/model33c_both'
#guiFile = '/data/ljo31/Lens/J1125/model33d_both'
#guiFile = '/data/ljo31/Lens/J1125/model33e_both'
#guiFile = '/data/ljo31/Lens/J1125/model33g_both'
#guiFile = '/data/ljo31/Lens/J1125/model33c_both'
#guiFile = '/data/ljo31/Lens/J1125/model33f_both'
#guiFile = '/data/ljo31/Lens/J1125/model23'
#guiFile = '/data/ljo31/Lens/J1125/model24'
guiFile = '/data/ljo31/Lens/J1125/gui_29b_C1'
guiFile = '/data/ljo31/Lens/J1125/gui34_A'

print guiFile

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
print img1[mask].shape # should be 6678

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
xoffset =  offsets[0][3]
yoffset = offsets[1][3]
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
            if key == 'pa':
                if val >0:
                    pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                elif val<0:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,0,value=val))
            elif key == 're':
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,60,value=val))
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
                if key == 're':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
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
        pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
        cov.append(s[key]['sdev'])
        p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=shear[0]['b']['value']))
cov.append(0.1)
p['b'] = pars[-1]
if shear[0]['pa']['value']> 0:
    pars.append(pymc.Uniform('extShear PA',0.,180.,value=shear[0]['pa']['value']))
elif shear[0]['pa']['value'] < 0:
    pars.append(pymc.Uniform('extShear PA',-180.,0,value=shear[0]['pa']['value']))
cov.append(10.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
           if key == 're':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
           elif key == 'n':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
           else:
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']) 
           elif key == 're':
               cov.append(s[key]['sdev']) 
           else:
               cov.append(s[key]['sdev'])
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            #print key, '= ', val
            lo,hi = lo - lenses[0].pars[key].value.item(), hi - lenses[0].pars[key].value.item()
            val = val - lenses[0].pars[key].value.item()
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi+5,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            #print p[key]
            cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            if key == 're':
                pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
            elif key == 'n':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            if key == 'pa':
                cov.append(s[key]['sdev']) 
            else:
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            lo,hi = lo - lenses[0].pars[key].value.item(), hi - lenses[0].pars[key].value.item()
            val = val - lenses[0].pars[key].value.item()
            print 'va', val, 'lo', lo, 'hi', hi
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi+5,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            cov.append(s[key]['sdev'])
    srcs.append(SBModels.Sersic(name,p))


#print len(pars), len(cov)
#for p in pars:
#    print p, p.value.item()

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

print len(cov), len(pars)

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=10,nwalkers=66,ntemps=4)
S.sample(1000)
outFile = '/data/ljo31/Lens/J1125/emcee'+str(X)
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


jj=0
for jj in range(20):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=10,nwalkers=66,ntemps=4,initialPars=trace[a1])
    S.sample(1000)

    outFile = '/data/ljo31/Lens/J1125/emcee'+str(X)+str(jj)
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

#S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nwalkers=100,nthreads=6) # should have 100 walkers.
#S.sample(1000)

result = S.result()
lp = result[0]

trace = numpy.array(result[1])
a1,a2,a3 = numpy.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)


## now we need to interpret these resultaeten
logp,coeffs,dic,vals = result
ii = np.where(logp==np.amax(logp))
#coeff = coeffs[ii][0]

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
pl.figure()
for i in range(4):
    pl.plot(lp[:,i,:])
pl.show()
