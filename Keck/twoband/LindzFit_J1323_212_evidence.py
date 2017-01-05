import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles
import emcee

X=2
print X
# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=560) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=560) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmax=500)
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmax=500)
    pl.title('signal-to-noise residuals')
    pl.colorbar()

image = py.open('/data/ljo31/Lens/J1323/J1323_nirc2.fits')[0].data.copy()[565:885,805:1140]/90.
sigma = np.ones(image.shape) 

result = np.load('/data/ljo31/Lens/LensModels/twoband/J1323_212')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

OVRS = 1
yc,xc = iT.overSample(image.shape,OVRS)
yo,xo = iT.overSample(image.shape,1)
xo,xc = xo-50,xc-50
yo,yc=yo-50,yc-50
xc,xo,yc,yo=xc*0.2,xo*0.2,yc*0.2,yo*0.2
xc,xo,yc,yo = xc+20,xo+20,yc+22,yo+22
mask = np.zeros(image.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

pars = []
cov = []
### 3 PSF components
pars.append(pymc.Uniform('xoffset',-40.,40.,value=5))
pars.append(pymc.Uniform('yoffset',-40.,40.,value=5))
cov += [0.4,0.4]
#psf1
pars.append(pymc.Uniform('sigma 1',0,8,value=3))
pars.append(pymc.Uniform('q 1',0.1,1,value=0.7))
pars.append(pymc.Uniform('pa 1',-180,180,value= 90 ))
pars.append(pymc.Uniform('amp 1',0,1,value=0.25))
cov += [1,0.5,50,0.5]

# psf2
pars.append(pymc.Uniform('sigma 2',0.1,60,value= 5 )) 
pars.append(pymc.Uniform('q 2',0.1,1,value=0.9))
pars.append(pymc.Uniform('pa 2',-180,180,value= 90 ))
pars.append(pymc.Uniform('amp 2',0,1,value=0.25))
cov += [1,0.5,50,0.5]

# psf3
pars.append(pymc.Uniform('sigma 3',0.1,400,value= 15 )) 
pars.append(pymc.Uniform('q 3',0.1,1,value=0.9))
pars.append(pymc.Uniform('pa 3',-180,180,value= 90 ))
pars.append(pymc.Uniform('amp 3',0,1,value=0.25))
cov += [1,0.5,50,0.5]

psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':pars[2],'q':pars[3],'pa':pars[4],'amp':pars[5]})
psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':pars[6],'q':pars[7],'pa':pars[8],'amp':pars[9]})
psfObj3 = SBObjects.Gauss('psf 3',{'x':0,'y':0,'sigma':pars[10],'q':pars[11],'pa':pars[12],'amp':pars[13]})
psfObjs = [psfObj1,psfObj2,psfObj3]

xpsf,ypsf = iT.coords((81,81))-40


gals = []
for name in ['Galaxy 1', 'Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

lenses = []
p = {}
for key in 'x','y','q','pa','b','eta':
    p[key] = dic['Lens 1 '+key][a1,a2,a3]
lenses.append(MassModels.PowerLaw('Lens 1',p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = dic['extShear'][a1,a2,a3]
p['pa'] = dic['extShear PA'][a1,a2,a3]
lenses.append(MassModels.ExtShear('shear',p))

los = dict([('q',0.05),('pa',-180.),('re',0.1),('n',0.1)])
#his = dict([('q',1.00),('pa',180.),('re',100.),('n',12.)])
covs = dict([('x',0.1),('y',0.1),('q',0.05),('pa',0.5),('re',0.5),('n',0.2)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',8.)])


srcs = []
for name in ['Source 2','Source 1']:
    p = {}
    if name == 'Source 2':
        for key in 'q','re','n','pa':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
        for key in 'x','y': 
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi=val-20,val+20
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1] #+ lenses[0].pars[key] 
            cov +=[5]
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
        for key in 'x','y': 
            p[key] = srcs[0].pars[key]
    srcs.append(SBBModels.Sersic(name,p))


npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]


@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    models = []
    dx = pars[0].value
    dy = pars[1].value 
    xp,yp = xc+dx,yc+dy
    psf = xpsf*0.
    for obj in psfObjs:
        obj.setPars()
        psf += obj.pixeval(xpsf,ypsf) / (np.pi*2.*obj.pars['sigma'].value**2.)
    psf = psf/np.sum(psf)
    psf = convolve.convolve(image,psf)[1]
    imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp[mask2] = gal.pixeval(xin,yin,0.2/OVRS,csub=23) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
    kk=0
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp[mask2] = src.pixeval(x0,y0,0.2/OVRS,csub=23)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    model[n] = -1*np.ones(model[n-1].shape)
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    model = (model.T*fit).sum(1)
    resid = (model-imin)/sigin
    lp = -0.5*(resid**2.).sum()
    return lp 
 
  
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

kresult = np.load('/data/ljo31/Lens/LensModels/J1323_Kp_212')#1323/twoband_Kp_212_2')
lp,trace,dic,_= kresult
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]

def prior(pars):
    logp = 0.
    for par in pars:
        logp += par.logp
    return logp

nwalkers, ndim, ntemps,nsteps = trace.shape[2], len(pars), trace.shape[1], trace.shape[0]
sampler = emcee.PTSampler(ntemps,nwalkers,ndim,likelihood,prior)

logL = lp.reshape((ntemps,nwalkers,nsteps)) - prior(pars)
ev = sampler.thermodynamic_integration_log_evidence(logL)
print ev
