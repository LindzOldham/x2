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

X=0

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
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=-5)
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=-10)
    pl.title('signal-to-noise residuals')
    pl.colorbar()

img1 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2_masked.fits')[0].data.copy() 
psf1 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1605/F814W_psf1new.fits')[0].data.copy()  
psf2 /= psf2.sum()
Dx,Dy = -15.,-15.

result = np.load('/data/ljo31/Lens/LensModels/J1605_212_final')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs=[]
for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,xo,yc,yo = xc+Dx , xo+Dx, yc+Dy, yo+Dy
mask = np.zeros(img1.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

los = dict([('q',0.03),('pa',-180.),('re',0.1),('n',0.1)])
his = dict([('q',1.00),('pa',180.),('re',150.),('n',10.)])
covs = dict([('x',0.2),('y',0.2),('q',0.05),('pa',0.5),('re',0.5),('n',0.1)])
lenslos = dict([('q',0.03),('pa',-180.),('b',0.1),('eta',0.5)]) 
lenshis = dict([('q',1.00),('pa',180.),('b',100.),('eta',1.5)])
covlens = dict([('x',0.2),('y',0.2),('q',0.05),('pa',0.5),('b',0.5),('eta',0.1)])

pars = []
cov = []
pars.append(pymc.Uniform('xoffset',-10.,10.,value=dic['xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('yoffset',-10.,10.,value=dic['yoffset'][a1,a2,a3]))
cov += [0.5,0.5]

gals = []
for name in ['Galaxy 1', 'Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'q','pa','re','n':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
        for key in 'x','y':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi=val-10,val+10
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1] #+ lenses[0].pars[key] 
            cov +=[2]            
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            val = dic[name+' '+key][a1,a2,a3]
            print key, val
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

lenses = []
for name in ['Lens 1']:
    p = {}
    for key in 'q','pa','b','eta':
        lo,hi,val = lenslos[key],lenshis[key],dic['Lens 1 '+key][a1,a2,a3]
        pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
        cov.append(covlens[key])
        p[key] = pars[-1]
    for key in 'x','y':
        val = dic[name+' '+key][a1,a2,a3]
        lo,hi=val-10,val+10
        pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
        cov.append(covlens[key])
        p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,a2,a3]))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180,value=dic['extShear PA'][a1,a2,a3]))
cov.append(10.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

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
            p[key] = pars[-1] + lenses[0].pars[key] 
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
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xc*0.
            tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=23) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=23)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        model[n] = np.ones(model[n].shape)
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

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=60,ntemps=3)
print 'set up sampler'
S.sample(500)
outFile = '/data/ljo31/Lens/J1605/finish_212_'+str(X)
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
for jj in range(20):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=60,ntemps=3,initialPars=trace[-1])
    S.sample(1000)

    outFile = '/data/ljo31/Lens/J1605/finish_212_'+str(X)
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
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=23) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp = src.pixeval(x0,y0,1./OVRS,csub=23)
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

