import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles

X='gri_B_2' # continuing gri_B_1, with slightly bigger image and better priors on re
print X

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.99) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.99) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-50,vmax=50,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()


img1 = py.open('/data/ljo31b/lenses/chip5/imgg.fits')[0].data[450:-450,420:-480]
bg=np.median(img1[-10:,-10:])
img1-=bg
sig1 = py.open('/data/ljo31b/lenses/chip5/noisemap_g_big.fits')[0].data
psf1 = py.open('/data/ljo31b/lenses/g_psf3.fits')[0].data 

img2 = py.open('/data/ljo31b/lenses/chip5/imgr.fits')[0].data[450:-450,420:-480]
bg=np.median(img2[-10:,-10:])
img2-=bg
sig2 = py.open('/data/ljo31b/lenses/chip5/noisemap_r_big.fits')[0].data
psf2 = py.open('/data/ljo31b/lenses/r_psf3.fits')[0].data

img3 = py.open('/data/ljo31b/lenses/chip5/imgi.fits')[0].data[450:-450,420:-480]
bg=np.median(img3[-10:,-10:])
img3-=bg
sig3 = py.open('/data/ljo31b/lenses/chip5/noisemap_i_big.fits')[0].data
psf3 = py.open('/data/ljo31b/lenses/i_psf3.fits')[0].data

py.writeto('/data/ljo31b/lenses/data/img1_bigger.fits',img1,clobber=True)

imgs = [img1,img2,img3]
sigs = [sig1,sig2,sig3]
psfs = [psf1,psf2,psf3]
PSFs = []
for i in range(len(psfs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

result = np.load('/data/ljo31b/lenses/model_gri_gri_B_1')
lp,trace,dic,_= result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xo,xc,yo,yc = xo-10,xc-10,yo-10,yc-10
mask = py.open('/data/ljo31b/lenses/chip5/newmask.fits')[0].data
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

pars = []
cov = []

# offsets
pars.append(pymc.Uniform('gr xoffset',-5.,5.,value=dic['gr xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('gr yoffset',-5.,5.,value=dic['gr yoffset'][a1,a2,a3]))
cov += [0.1,0.1]
pars.append(pymc.Uniform('gi xoffset',-10.,10.,value=dic['gi xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('gi yoffset',-10.,10.,value=dic['gi yoffset'][a1,a2,a3]))
cov += [0.1,0.1]

los = dict([('q',0.05),('pa',-180.),('re',0.01),('n',0.5)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',10.)])
covs = dict([('x',0.1),('y',0.1),('q',0.05),('pa',0.5),('re',0.1),('n',0.1)])
covlens = dict([('x',0.1),('y',0.1),('q',0.05),('pa',0.5),('b',0.1),('eta',0.1)])
lenslos, lenshis = dict([('q',0.05),('pa',-180.),('b',0.5),('eta',0.5)]), dict([('q',1.00),('pa',180.),('b',100.),('eta',1.5)])
gals = []
for name in ['Galaxy 1', 'Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            val = dic[name+' '+key][a1,a2,a3]
            if key == 'x' or key == 'y':
                lo,hi=val-10,val+10
            else:
                lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
    elif name == 'Galaxy 2':
        for key in 'x','y','q','pa','re','n':
            val = dic[name+' '+key][a1,a2,a3]
            if key == 'x' or key == 'y':
                lo,hi=val-10,val+10
            else:
                lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
    gals.append(SBModels.Sersic(name,p))


lenses = []
name = 'Lens 1'
p = {}
for key in 'x','y','q','pa','b','eta':
    val = dic['Lens 1 '+key][a1,a2,a3]
    if key  in ['x','y']:
        lo,hi=val-5,val+5
    else:
        lo,hi= lenslos[key],lenshis[key]
    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
    p[key] = pars[-1]
    cov.append(covlens[key])
lenses.append(MassModels.PowerLaw('Lens 1',p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,a2,a3]))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180,value=dic['extShear PA'][a1,a2,a3]))
cov.append(1.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
name = 'Source 1'
p = {}
for key in 'q','re','n','pa':
    val = dic[name+' '+key][a1,a2,a3]
    lo,hi= los[key],his[key]
    pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
    p[key] = pars[-1]
    cov.append(covs[key])
for key in 'x','y': 
    val = dic[name+' '+key][a1,a2,a3]
    lo,hi=val-10,val+10
    pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
    p[key] = pars[-1] + lenses[0].pars[key]
    cov.append(covs[key])
srcs.append(SBBModels.Sersic(name,p))
name = 'Source 2'
p = {}
for key in 'x','y','q','re','n','pa':
    val = dic[name+' '+key][a1,a2,a3]
    if key=='x' or key=='y':
        lo,hi=val-10,val+10
    else:
        lo,hi= los[key],his[key]
    pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
    p[key] = pars[-1]
    cov.append(covs[key])
srcs.append(SBBModels.Sersic(name,p))


@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    models = []
    for i in range(len(imgs)):
        if i == 0:
            dx,dy = 0,0
        elif i == 1:
            dx = pars[0].value 
            dy = pars[1].value 
        elif i == 2:
            dx = pars[2].value 
            dy = pars[3].value 
        else:
            print 'ERROR'
        xp,yp = xc+dx,yc+dy
        image,sigma,psf = imgs[i],sigs[i],PSFs[i]
        imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
        n = 0
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xc*0.
            tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=21) 
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
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=21)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
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


S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=8,nwalkers=120,ntemps=3,initialPars=trace[-1])
S.sample(500)
outFile = '/data/ljo31b/lenses/model_gri_'+str(X)
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
#result = np.load('/data/ljo31b/lenses/model_gri_'+str(X))
lp = result[0]
trace = numpy.array(result[1])
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

jj=0
for jj in range(30):
    S.p0 = trace[-1]
    print 'sampling'
    S.sample(1000)

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


colours = ['g','r','i']
models = []
fits = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    elif i == 1:
        dx = pars[0].value 
        dy = pars[1].value 
    elif i == 2:
        dx = pars[2].value 
        dy = pars[3].value 
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
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=21) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp = src.pixeval(x0,y0,1./OVRS,csub=21)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    rhs = image[mask]/sigma[mask]
    print model.shape, model.size
    mmodel = model.reshape((n,image.shape[0],image.shape[1]))
    mmmodel = np.empty(((len(gals) + len(srcs)),image[mask].size))
    for m in range(mmodel.shape[0]):
        print mmodel[m].shape
        mmmodel[m] = mmodel[m][mask]
    op = (mmmodel/sigma[mask]).T
    rhs = image[mask]/sigma[mask]
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
       
pl.figure()
pl.plot(lp[:,0])
pl.show()

from tools.simple import climshow
for i in range(len(components)):
    pl.figure()
    climshow(components[i])
    pl.colorbar()

pl.show()
py.writeto('/data/ljo31b/lenses/model_B_0_ctd.fits',image-model)
