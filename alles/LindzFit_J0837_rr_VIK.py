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

'''
X=0 - model5_v. Lens ended up moving down to a ridiculous place, dragging source with it.
X=1 - trying again, chaining lens to galaxy.
X=1 - model4_i. X=1 went REALLY weird. (overwrote the last one, sorry)
X=2 - as above, but with the lens position independent again.
X=4 - Model3
X=5 - as 4, but using mask7 for the dust lane. Still a 2/1/1 model (could try 2/1/2 later on!)
X=6 - not using mask7 or anything, but using the dustlanenew sig files TO RUN
X=7 - as 6, but starting off from the end of X=4 to save the walkers' efforts RUNNING
X=8 - fitting for a dustlane! A negative, disky galaxy component.
'''
print 'ici'
X =0
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

img1 = py.open('/data/ljo31/Lens/J0837/F606W_sci_cutout.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]
sig1 = py.open('/data/ljo31/Lens/J0837/F606W_noisemap.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]
psf1 = py.open('/data/ljo31/Lens/J0837/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J0837/F814W_sci_cutout.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]
sig2 = py.open('/data/ljo31/Lens/J0837/F814W_noisemap.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]
psf2 = py.open('/data/ljo31/Lens/J0837/F814W_psf3.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)

img3 = py.open('/data/ljo31/Lens/J0837/J0837_Kp_narrow_med.fits')[0].data.copy()[810:1100,790:1105]    #[790:1170,700:1205]
sig3 = np.ones(img3.shape) 


guiFile = '/data/ljo31/Lens/J0837/Model3'

print guiFile

result = np.load('/data/ljo31/Lens/J0837/emcee8')

lp= result[0]
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

imgs = [img1,img2,img3]
sigs = [sig1,sig2,sig3]
psfs = [psf1,psf2]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,xo=xc-30.,xo-30.
yc,yo=yc-20.,yo-20.
yck,xck = iT.overSample(img3.shape,OVRS)
yok,xok = iT.overSample(img3.shape,1)
xck,xok,yck,yok=xck*0.2,xok*0.2,yck*0.2,yok*0.2
xck,xok = xck+16 , xok+16 # check offsets
yck,yok = yck+23 , yok+23 # 
mask = py.open('/data/ljo31/Lens/J0837/mask.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15] #+ py.open('/data/ljo31/Lens/J0837/mask7.fits')[0].data.copy()
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(yc,xc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0
mask3,mask32 = np.ones(img3.shape), np.ones(xck.shape)
mask3,mask32=mask3==1, mask32==1
masks = [mask,mask,mask3]
mask2s = [mask2,mask2,mask32]

for i in range(len(imgs[:-1])):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)
PSFs.append([0])

G,L,S,offsets,shear = numpy.load(guiFile)

### offset between V and I; offset between V and Kp
pars = []
cov = []
pars.append(pymc.Uniform('xoffset',-5.,5.,value=dic['xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=dic['yoffset'][a1,a2,a3]))
cov += [0.2,0.2]
pars.append(pymc.Uniform('Kxoffset',-10.,10.,value=0))
pars.append(pymc.Uniform('Kyoffset',-10.,10.,value=0))
cov += [0.4,0.4]

gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))


lenses = []
for name in L.keys():
    s = L[name]
    p = {}
    for key in 'x','y','q','pa','b','eta':
        lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
        pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
        if key=='pa':
            cov.append(s[key]['sdev']*100)
        else:
            cov.append(s[key]['sdev']*10)
        p[key] = pars[-1]
    #for key in 'x','y':
    #    p[key] = gals[0].pars[key]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,a2,a3]))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180,value=dic['extShear PA'][a1,a2,a3]))
cov.append(100.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))
scales = dict([('x',5.),('y',,5.),('q',0.1),('pa',10.),('re',1.),('n',0.5)]
srcs = []
for name in ['Source 1']:
    s = S[name]
    p = {}
    if name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
           pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']*100) 
           else:
               cov.append(s[key]['sdev']*10)
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]
            lo,hi = lo - lenses[0].pars[key].value.item(), hi - lenses[0].pars[key].value.item()
            #val = val - lenses[0].pars[key].value.item()
            pars.append(pymc.Uniform('%s %s'%(name,key),lo-1 ,hi+1,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            cov.append(s[key]['sdev'])
    srcs.append(SBModels.Sersic(name,p))
name='Source 2'
for key in 'x','y','q','re','n','pa':
    lo,hi,val = s[key]['lower'],s[key]['upper'],dic[name+' '+key][a1,a2,a3]#s[key]['value']
    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
    p[key] = pars[-1]
    if key == 'pa':
        cov.append(s[key]['sdev']*100) 
    else:
        cov.append(s[key]['sdev']*10)
srcs.append(SBModels.Sersic(name,p))



### keck params - a 2-component Gaussian
pars.append(pymc.Uniform('sigma 1',0,8,value=4))
cov += [5]
pars.append(pymc.Uniform('q 1',0,1,value=0.7))
cov += [1]
pars.append(pymc.Uniform('pa 1',-180,180,value= 90 ))
cov += [50]
pars.append(pymc.Uniform('amp 1',0,1,value=0.7))
cov += [4]
pars.append(pymc.Uniform('sigma 2',10,60,value= 20 )) 
cov += [40]
pars.append(pymc.Uniform('q 2',0,1,value=0.9))
cov += [1]
pars.append(pymc.Uniform('pa 2',-180,180,value= 90 ))
cov += [50]

xpsf,ypsf = iT.coords((340,340))-170

print srcs[1].pars
print srcs[0].pars

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
            xp,yp = xc+dx,yc+dy
        elif i == 1:
            dx = pars[0].value 
            dy = pars[1].value 
            xp,yp = xc+dx,yc+dy
        elif i == 2:
            dx = pars[0].value + pars[2].value
            dy = pars[1].value + pars[3].value
            xp,yp = xck+dx,yck+dy
            sigma1 = pars[-7].value.item()
            q1 = pars[-6].value.item()
            pa1 = pars[-5].value.item()
            amp1 = pars[-4].value.item()
            sigma2 = pars[-3].value.item()
            q2 = pars[-2].value.item()
            pa2 = pars[-1].value.item()
            amp2 = 1.-amp1
            psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':sigma1,'q':q1,'pa':pa1,'amp':10})
            psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':sigma2,'q':q2,'pa':pa2,'amp':10})
            psf1 = psfObj1.pixeval(xpsf,ypsf) * amp1 / (np.pi*2.*sigma1**2.)
            psf2 = psfObj2.pixeval(xpsf,ypsf) * amp2 / (np.pi*2.*sigma2**2.)
            psf = psf1 + psf2
            psf /= psf.sum()
            psf = convolve.convolve(img3,psf)[1]
            PSFs[2] = psf
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        mask,mask2=masks[i],mask2s[i]
        imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
        n = 0
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xp*0.
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
            tmp = xp*0.
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=11)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            if src.name=='Source 2':
                model[n] *=-1
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

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=82,ntemps=3)
S.sample(1000)
outFile = '/data/ljo31/Lens/J0837/emcee_VIK'+str(X)
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
for jj in range(16):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=82,ntemps=3,initialPars=trace[a1])
    S.sample(1000)

    outFile = '/data/ljo31/Lens/J0837/emcee_VIK'+str(X)
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




colours = ['V', 'I', 'K']
#mods = S.blobs
models = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
        xp,yp = xc+dx,yc+dy
        xop,xop=xo+dx,yo+dy
    elif i == 1:
        dx = pars[0].value 
        dy = pars[1].value 
        xp,yp = xc+dx,yc+dy
        xop,xop=xo+dx,yo+dy
    elif i == 2:
        dx = pars[0].value + pars[2].value
        dy = pars[1].value + pars[3].value
        xp,yp = xck+dx,yck+dy
        xop,xop=xok+dx,yok+dy
        sigma1 = pars[-7].value.item()
        q1 = pars[-6].value.item()
        pa1 = pars[-5].value.item()
        amp1 = pars[-4].value.item()
        sigma2 = pars[-3].value.item()
        q2 = pars[-2].value.item()
        pa2 = pars[-1].value.item()
        amp2 = 1.-amp1
        psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':sigma1,'q':q1,'pa':pa1,'amp':10})
        psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':sigma2,'q':q2,'pa':pa2,'amp':10})
        psf1 = psfObj1.pixeval(xpsf,ypsf) * amp1 / (np.pi*2.*sigma1**2.)
        psf2 = psfObj2.pixeval(xpsf,ypsf) * amp2 / (np.pi*2.*sigma2**2.)
        psf = psf1 + psf2
        psf /= psf.sum()
        psf = convolve.convolve(img3,psf)[1]
        PSFs[2] = psf
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xp*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xp*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=11)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        if src.name == 'Source 2':
            model[n] *=-1
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
    comps = False
    if comps == True:
        for i in range(len(gals)+len(srcs)):
            pl.figure()
            pl.imshow(components[i],interpolation='nearest',origin='lower',cmap='afmhot')
            pl.colorbar()



