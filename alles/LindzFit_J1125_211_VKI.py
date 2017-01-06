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
print X

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=2) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=2) #,vmin=vmin,vmax=vmax)
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


img3 = py.open('/data/ljo31/Lens/J1125/Kp_J1125_nirc2_n.fits')[0].data.copy()[650:905,640:915]
sig3 = np.ones(img3.shape)

imgs = [img1,img2,img3]
sigs = [sig1,sig2,sig3]
psfs = [psf1,psf2]
PSFs = []
for i in range(len(psfs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

scales = [1.,1.,0.2]

result = np.load('/data/ljo31/Lens/LensModels/J1125_211')
lp,trace,dic,_= result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
kresult = np.load('/data/ljo31/Lens/J1125/KeckPSF__final')
klp,ktrace,kdic,_= kresult
ka2=0
ka1,ka3 = numpy.unravel_index(klp[:,0].argmax(),klp[:,0].shape)

OVRS = 1
yck,xck = iT.overSample(img3.shape,OVRS)
yok,xok = iT.overSample(img3.shape,1)
xck,xok,yck,yok=xck*0.2,xok*0.2,yck*0.2,yok*0.2
xck,xok,yck,yok = xck+10,xok+10,yck+12,yok+12
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
mask = np.zeros(img1.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

pars = []
cov = []

### first parameters need to be the offsets
pars.append(pymc.Uniform('xoffset',-5.,5.,value=dic['xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=dic['yoffset'][a1,a2,a3]))
cov += [0.4,0.4]
pars.append(pymc.Uniform('k xoffset',-15.,15.,value=kdic['xoffset'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('k yoffset',-15.,15.,value=kdic['yoffset'][ka1,ka2,ka3]))
cov += [0.4,0.4]

### four PSF components
#psf1
pars.append(pymc.Uniform('sigma 1',0.1,8,value=kdic['sigma 1'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('q 1',0,1,value=kdic['q 1'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('pa 1',-180,180,value= kdic['pa 1'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('amp 1',0,1,value=kdic['amp 1'][ka1,ka2,ka3]))
cov += [1,0.5,50,0.5]
# psf2
pars.append(pymc.Uniform('sigma 2',0.1,60,value=kdic['sigma 2'][ka1,ka2,ka3]  )) 
pars.append(pymc.Uniform('q 2',0,1,value=kdic['q 2'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('pa 2',-180,180,value= kdic['pa 2'][ka1,ka2,ka3] ))
pars.append(pymc.Uniform('amp 2',0,1,value=kdic['amp 2'][ka1,ka2,ka3]))
cov += [1,0.5,50,0.2]
# psf3
pars.append(pymc.Uniform('sigma 3',0.1,400,value=kdic['sigma 3'][ka1,ka2,ka3] )) 
pars.append(pymc.Uniform('q 3',0,1,value=kdic['q 3'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('pa 3',-180,180,value= kdic['pa 3'][ka1,ka2,ka3]))
pars.append(pymc.Uniform('amp 3',0,1,value=kdic['amp 3'][ka1,ka2,ka3]))
cov += [1,0.5,50,0.2]
# psf4
#pars.append(pymc.Uniform('sigma 4',0.1,400,value=kdic['sigma 4'][ka1,ka2,ka3]))
#pars.append(pymc.Uniform('q 4',0,1,value=kdic['q 4'][ka1,ka2,ka3]))
#pars.append(pymc.Uniform('pa 4',-180,180,value= kdic['pa 4'][ka1,ka2,ka3] ))
#pars.append(pymc.Uniform('amp 4',0,1,value=kdic['amp 4'][ka1,ka2,ka3]))
#cov += [1,0.5,50,0.2]

psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':pars[4],'q':pars[5],'pa':pars[6],'amp':pars[7]})
psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':pars[8],'q':pars[9],'pa':pars[10],'amp':pars[11]})
psfObj3 = SBObjects.Gauss('psf 3',{'x':0,'y':0,'sigma':pars[12],'q':pars[13],'pa':pars[14],'amp':pars[15]})
#psfObj4 = SBObjects.Gauss('psf 4',{'x':0,'y':0,'sigma':pars[16],'q':pars[17],'pa':pars[18],'amp':pars[19]})

psfObjs = [psfObj1,psfObj2,psfObj3]#,psfObj4]

xpsf,ypsf = iT.coords((81,81))-40
los = dict([('q',0.05),('pa',-180.),('re',0.1),('n',0.5)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',10.)])
covs = dict([('x',0.1),('y',0.1),('q',0.05),('pa',10.),('re',3.),('n',0.2)])
covlens = dict([('x',0.1),('y',0.1),('q',0.05),('pa',10.),('b',0.2),('eta',0.1)])
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
        for key in 'q','pa','re','n':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
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

'''
lenses = []
p = {}
for key in 'x','y','q','pa','b','eta':
    val = dic['Lens 1 '+key][a1,a2,a3]
    if key  == 'x' or 'y':
        lo,hi=val-10,val+10
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
cov.append(10.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))'''

srcs = []
for name in ['Source 1']:
    p = {}
    if name == 'Source 1':
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
    elif name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1]
            cov.append(covs[key])
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
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
        image, sigma,scale = imgs[i],sigs[i],scales[i]
        if i == 0:
            dx,dy = 0,0
            xp,yp = xc+dx,yc+dy
            psf = PSFs[i]
        elif i ==1:
            dx = pars[0].value 
            dy = pars[1].value 
            xp,yp = xc+dx,yc+dy
            psf = PSFs[i]
        elif i ==2:
            dx = pars[2].value 
            dy = pars[3].value
            xp,yp = xck+dx,yck+dy
            psf = xpsf*0.
            for obj in psfObjs:
                obj.setPars()
                psf += obj.pixeval(xpsf,ypsf) / (np.pi*2.*obj.pars['sigma'].value**2.)
            psf = psf/psf.sum()
            psf = convolve.convolve(image,psf)[1]
        mask,mask2=np.ones(image.shape),np.ones(image.shape)
        mask,mask2=mask==1,mask2==1
        imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
        n = 0
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            #print gal.pars['x'].value, gal.pars['y'].value,gal.pars['q'].value, gal.pars['pa'].value,gal.pars['re'].value, gal.pars['n'].value
            tmp = xp*0.
            tmp[mask2] = gal.pixeval(xin,yin,scale/OVRS,csub=1) 
            tmp = iT.resamp(tmp,OVRS,True) 
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        for lens in lenses:
            lens.setPars()
        x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
        for src in srcs:
            src.setPars()
            #print src.pars['x'].value, src.pars['y'].value,src.pars['q'].value, src.pars['pa'].value,src.pars['re'].value, src.pars['n'].value
            tmp = xp*0.
            tmp[mask2] = src.pixeval(x0,y0,scale/OVRS,csub=1)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        model[n] = np.ones(model[n-1].shape)
        rhs = (imin/sigin) 
        op = (model/sigin).T 
        fit, chi = optimize.nnls(op,rhs)
        model = (model.T*fit).sum(1)
        resid = (model-imin)/sigin
        lp += -0.5*(resid**2.).sum()
        models.append(model)
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


S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=22,nwalkers=64,ntemps=3)
S.sample(500)
outFile = '/data/ljo31/Lens/J1125/VKI_211_'+str(X)
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
for jj in range(30):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=22,nwalkers=64,ntemps=3,initialPars=trace[-1])
    S.sample(500)

    outFile = '/data/ljo31/Lens/J1125/VKI_211_'+str(X)
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


colours=['V','I','K']
models=[]
for i in range(len(imgs)):
    image, sigma,scale = imgs[i],sigs[i],scales[i]
    if i == 0:
        dx,dy = 0,0
        xp,yp = xc+dx,yc+dy
        psf = PSFs[i]
    elif i ==1:
        dx = pars[0].value 
        dy = pars[1].value 
        xp,yp = xc+dx,yc+dy
        psf = PSFs[i]
    elif i ==2:
        dx = pars[2].value 
        dy = pars[3].value
        xp,yp = xck+dx,yck+dy
        psf = xpsf*0.
        for obj in psfObjs:
            obj.setPars()
            psf += obj.pixeval(xpsf,ypsf) / (np.pi*2.*obj.pars['sigma'].value**2.)
        psf = psf/psf.sum()
        psf = convolve.convolve(image,psf)[1]
    mask,mask2=np.ones(image.shape),np.ones(image.shape)
    mask,mask2=mask==1,mask2==1
    imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xp*0.
        tmp[mask2] = gal.pixeval(xin,yin,scale/OVRS,csub=1) 
        tmp = iT.resamp(tmp,OVRS,True) 
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xp*0.
        tmp[mask2] = src.pixeval(x0,y0,scale/OVRS,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    model[n] = np.ones(model[n-1].shape)
    n+=1
    rhs = (imin/sigin) 
    op = (model/sigin).T 
    fit, chi = optimize.nnls(op,rhs)
    print image.shape
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
