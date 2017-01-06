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

X=0
X=1 # no dustlane, psf1neu_start8
X=2 # psf1neu_cond1 - subtracting pedestal from psfs. OVRS = 1 to speed things up
X = 5 # dustlane, psf1neu_cond1
X = 6 # ovrs=3 and a dust lane
X=10 # 1/1/2, ovrs=2
X = 11 # running on calx079: 112model, could be promising
X = 12 # awry1 - changing psf and sig2*2


X = 0# 112model_v_psf1new - which is actually a 212 out of sheer desperation
X = 1 # 112modl_v_psf2new, ovrs=3,flat noisemap
X=3 # noisemap times 2, otherwise same as 1
X = 6 # BIG, ovrs=3,112model_v_psf2new
X = 7 # as 6, but with 2*sig1
X = 9 # gui7both_c
X = 11 # as 9, but with sources concentric
print X
''' new!!! '''

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



img1 = py.open('/data/ljo31/Lens/J1619/F606W_sci_cutout_big.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1619/F606W_noisemap_big.fits')[0].data.copy()*2.
psf1 = py.open('/data/ljo31/Lens/J1619/F606W_psf2new.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)


img2 = py.open('/data/ljo31/Lens/J1619/F814W_sci_cutout_big.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1619/F814W_noisemap_big.fits')[0].data.copy()
sig2[sig2>0.07] = 0.07
sig2 *= 2.
psf2 = py.open('/data/ljo31/Lens/J1619/F814W_psf1neu.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)
psf3 = psf2.copy()
psf3[5:-5,5:-5] = np.nan
cond = np.isfinite(psf3)
m = psf3[cond].mean()
psf2 = psf2 - m
psf2 = psf2/np.sum(psf2)

guiFile = '/data/ljo31/Lens/J1619/imdb2'
#guiFile = '/data/ljo31/Lens/J1619/psf1neu_start8'
guiFile = '/data/ljo31/Lens/J1619/awry1'
guiFile = '/data/ljo31/Lens/J1619/112model_v_psf1new' # this actually is 212
guiFile = '/data/ljo31/Lens/J1619/112model_v_psf2new'
guiFile = '/data/ljo31/Lens/J1619/gui7both_c'

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,xo = xc-35,xo-35
yc,yo=yc-40,yo-40

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

G,L,S,offsets,shear = numpy.load(guiFile)
guishear = shear[0].copy()

pars = []
cov = []
### first parameters need to be the offsets
pars.append(pymc.Uniform('xoffset',-5.,5.,value=offsets[0][3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=offsets[1][3]))
cov += [0.4,0.4]

gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*1)
        for key in 'x','y':
            p[key]=gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))


lenses = []
for name in L.keys():
    s = L[name]
    p = {}
    for key in 'x','y','q','pa','b','eta':
        lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
        if key == 'eta':
            hi = 1.7
        pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
        cov.append(s[key]['sdev']*1)
        p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=shear[0]['b']['value']))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180,value=shear[0]['pa']['value']))
cov.append(10.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
        for key in 'x','y': 
            #lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            #lo,hi = lo - lenses[0].pars[key], hi - lenses[0].pars[key]
            #val = val - lenses[0].pars[key]
            #pars.append(pymc.Uniform('%s %s'%(name,key),lo-1 ,hi+1,value=val ))  
            #p[key] = pars[-1] + lenses[0].pars[key]
            #cov.append(s[key]['sdev'])
            p[key] = srcs[0].pars[key]
    elif name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
        for key in 'x','y':
            lo,hi,val = S['Source 1'][key]['lower'],S['Source 1'][key]['upper'],S['Source 1'][key]['value']
            lo,hi = lo - lenses[0].pars[key], hi - lenses[0].pars[key]
            val = val - lenses[0].pars[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo-1 ,hi+1,value=val ))  
            p[key] = pars[-1] + lenses[0].pars[key]
            cov.append(s[key]['sdev'])
    srcs.append(SBModels.Sersic(name,p))

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
        imin,sigin,xin,yin = image.flatten(),sigma.flatten(),xp.flatten(),yp.flatten()
        n = 0
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xp*0.
            tmp = gal.pixeval(xin,yin,1./OVRS,csub=23).reshape(xp.shape)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp.ravel()
            n +=1
        for lens in lenses:
            lens.setPars()
        x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
        for src in srcs:
            src.setPars()
            tmp = xp*0.
            tmp = src.pixeval(x0,y0,1./OVRS,csub=23).reshape(xp.shape)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp.ravel()
            n +=1
        model[n] = np.ones(model[n-1].size)
        n+=1
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
    return lp

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)

print len(cov), len(pars)
import time
start=time.time()
S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=60,ntemps=3)
S.sample(1000)
print time.time()-start

outFile = '/data/ljo31/Lens/J1619/gehtschon_oneband_'+str(X)
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp = result[0]
trace = numpy.array(result[1])
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

jj=0
for jj in range(30):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=60,ntemps=3,initialPars=trace[-1])
    S.sample(1000)

    outFile = '/data/ljo31/Lens/J1619/gehtschon_oneband_'+str(X)
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()

    result = S.result()
    lp = result[0]
    trace = numpy.array(result[1])
    dic = result[2]
    a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,a3,i]
    print jj
    jj+=1


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
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xp*0.
        tmp = gal.pixeval(xin,yin,1./OVRS,csub=23).reshape(xp.shape)
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xp*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=23).reshape(xp.shape)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        if src.name == 'Source 2':
            model[n] *= -1
        n +=1
    model[n] = np.ones(model[n-1].shape)
    n+=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    #for i in range(4):
    #    pl.figure()
    #    pl.imshow(components[i],origin='lower',interpolation='nearest')
    model = components.sum(0)
    NotPlicely(image,model,sigma)
    
