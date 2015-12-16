import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline

'''
X=0 - TO RUN
'''
X=0
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

img = py.open('/data/ljo31/Lens/J0837/J0837_Kp_narrow_med.fits')[0].data.copy()[810:1100,790:1105]    #[790:1170,700:1205]
sig = np.ones(img.shape) 
psf = py.open('/data/ljo31/Lens/J0837/PSF_nirc2_Kp_narrow.fits')[0].data.copy()[25:-25,25:-25]
psf = psf/np.sum(psf)

result = np.load('/data/ljo31/Lens/J0837/emcee8')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

guiFile = '/data/ljo31/Lens/J0837/'
G,L,S,offsets,shear = numpy.load(guiFile)


imgs = [img]
sigs = [sig]
psfs = [psf]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img.shape,OVRS)
yo,xo = iT.overSample(img.shape,1)
xc,xo,yc,yo=xc*0.2,xo*0.2,yc*0.2,yo*0.2
xc,xo = xc+16 , xo+16 # check offsets
yc,yo = yc+23 , yo+23 # 
print np.mean(yo),np.mean(xo)
mask = np.zeros(img.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(yc,xc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
print mask2.shape, mask.shape
mask2 = mask2==0
mask = mask==0
print img[mask].size

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

covs = []
pars = []
pars.append(pymc.Uniform('xoffset',-10.,10.,value=0))
pars.append(pymc.Uniform('yoffset',-10.,10.,value=0))
cov += [0.4,0.4]

gals = []
for name in ['Galaxy 1', 'Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y':
            lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'x','y':
            p[key] = gals[0].pars[key]
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    gals.append(SBBModels.Sersic(name,p))

lenses = []
p = {}
for key in 'x','y':
    lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
    p[key] = pars[-1]
    cov.append(s[key]['sdev']*10)
for key in 'q','pa','b','eta':
    p[key] = dic['Lens 1 '+key][a1,a2,a3]
lenses.append(MassModels.PowerLaw('Lens 1',p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = dic['extShear'][a1,a2,a3]
p['pa'] = dic['extShear PA'][a1,a2,a3]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in ['Source 2', 'Source 1']:
    p = {}
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]+lenses[0].pars[key]
            cov.append(s[key]['sdev']*10)
    srcs.append(SBBModels.Sersic(name,p))




npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    models = []
    for i in range(len(imgs)):
        dx = pars[0].value 
        dy = pars[1].value 
        xp,yp = xc+dx,yc+dy
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
        n = 0
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xc*0.
            tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
            tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        for lens in lenses:
            lens.setPars()
        x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
        kk = 0
        for src in srcs:
            src.setPars()
            tmp = xc*0.
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=11)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            if kk == 1:
                model[n] *= -1
            kk +=1
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
    return lp #[0]

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)

print len(cov), len(pars)

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=5,nwalkers=20,ntemps=3)
S.sample(500)
outFile = '/data/ljo31/Lens/J0837/emcee_Kp_PSF_'+str(X)
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp = result[0]
trace = numpy.array(result[1])
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

jj=0
for jj in range(10):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=5,nwalkers=20,ntemps=3,initialPars=trace[a1])
    S.sample(500)

    outFile = '/data/ljo31/Lens/J0837/emcee_Kp_PSF_'+str(X)
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()

    result = S.result()
    lp = result[0]

    trace = numpy.array(result[1])
    a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,a3,i]
    print jj
    jj+=1


