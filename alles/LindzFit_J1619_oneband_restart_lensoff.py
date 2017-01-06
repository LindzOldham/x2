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

X = 11

'''
X=0 - model3
X=1 - Vmodel5, Vband
X=2 - Vmodel5, Vband, bigger image. Testing to see what effect the use of a bigger image has, if any, by doing exactly the same procedure as X=1.
X=3 - 1_Iband. Put X=1 bak into gui and now on I band.
X=4 - 1_both4 - put X=1 onto both bands in the gui.
X = 5 - 1both5
X=6 - 2_Iband2

'''

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
sig1 = py.open('/data/ljo31/Lens/J1619/F606W_noisemap_big.fits')[0].data.copy()
psf1 = py.open('/data/ljo31/Lens/J1619/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J1619/F814W_sci_cutout_big.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1619/F814W_noisemap_big.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1619/F814W_psf1.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)

guiFile = '/data/ljo31/Lens/J1619/1_Iband'
result = np.load('/data/ljo31/Lens/J1619/emcee_restart3')
lp= result[0]
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
trace = result[1]
dic = result[2]


print guiFile

imgs = [img2]
sigs = [sig2]
psfs = [psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xo,xc=xo-35,xc-35
yo,yc=yo-40,yc-40
print np.mean(xo), np.mean(yo)
mask_V = py.open('/data/ljo31/Lens/J1619/mask.fits')[0].data.copy()#[10:-10,10:-10]
mask_I = mask_V.copy()
startmasks = [mask_V, mask_I]
masks,mask2s = [], []
for mask in startmasks:
    tck = RectBivariateSpline(yo[:,0],xo[0],mask)
    mask2 = tck.ev(xc,yc)
    mask2[mask2<0.5] = 0
    mask2[mask2>0.5] = 1
    mask2 = mask2==0
    mask = mask==0
    masks.append(mask)
    mask2s.append(mask2)

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
gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev']*10)
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
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
        val = dic[name+' '+key][a1,a2,a3]
        p[key] = val
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = dic['extShear'][a1,a2,a3]
p['pa'] = dic['extShear PA'][a1,a2,a3]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
           val = dic[name+' '+key][a1,a2,a3]
           if key == 'n':
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,10,value=val))
           else:
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']*100) 
           elif key == 're':
               cov.append(s[key]['sdev']*10) 
           else:
               cov.append(s[key]['sdev']*10)
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi = lo - lenses[0].pars[key], hi - lenses[0].pars[key]
            #val = val - lenses[0].pars[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo-1 ,hi+1,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            print p[key]
            cov.append(s[key]['sdev'])
    elif name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            if key == 'pa':
                cov.append(s[key]['sdev']*100) 
            else:
                cov.append(s[key]['sdev']*10)
        for key in 'x','y':
            #lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            #lo,hi = lo - lenses[0].pars[key].value.item(), hi - lenses[0].pars[key].value.item()
            #val = val - lenses[0].pars[key].value.item()
            #pars.append(pymc.Uniform('%s %s'%(name,key),lo-1 ,hi+1,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            #p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            #cov.append(s[key]['sdev'])
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

# calculate x0,y0 for eachband before going into the inference
coords = []
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
    mask = masks[i]
    mask2 = mask2s[i]
    imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
    coords.append([x0,y0])

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
        mask = masks[i]
        mask2 = mask2s[i]
        x0,y0=coords[i]
        imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
        n = 0
        model = np.empty(((len(gals) + len(srcs)+1),imin.size))
        for gal in gals:
            gal.setPars()
            tmp = xc*0.
            tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
            tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        for src in srcs:
            src.setPars()
            tmp = xc*0.
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=1)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        model[n] = np.ones(model[n-1].shape)
        rhs = (imin/sigin) # data
        op = (model/sigin).T # model matrix
        fit, chi = optimize.nnls(op,rhs)
        #print fit
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

starter=np.zeros((4,72,16))
starter[:,:,:10] = trace[-1,:,:,:10]
starter[:,:,10:] = trace[-1,:,:,18:]

print len(cov), len(pars)
print trace[a1].shape
print starter.shape
import time
start=time.time()
S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=72,ntemps=4,initialPars=starter)
S.sample(1000)
print time.time()-start

outFile = '/data/ljo31/Lens/J1619/emcee_restart'+str(X)
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
for jj in range(10):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=72,ntemps=4,initialPars=trace[-1])
    S.sample(1000)

    outFile = '/data/ljo31/Lens/J1619/emcee_restart'+str(X)
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



## now we need to interpret these resultaeten
logp,coeffs,dic,vals = result
ii = np.where(logp==np.amax(logp))
#coeff = coeffs[ii][0]


# now showing a fit using the MASKED image
colours = ['F555W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
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
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
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
    model[n]=np.ones(model[n-1].shape)
    n+=1
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
for i in range(4):
    pl.plot(lp[:,i,:])
pl.show()

## update dictionary so we can read things back again
for name in G.keys():
    s = G[name]
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
        for key in 'x','y':
            s[key]['value'] = np.array(dic['Galaxy 1 '+key][a1,a2,a3])

for name in L.keys():
    s = L[name]
    for key in 'x','y','q','pa','b','eta':
        s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])

for name in guishear.keys():
    guishear[name]['value'] = np.array(dic['Lens 1 '+key][a1,a2,a3])

guishear['b']['value'] = np.array(dic['extShear'][a1,a2,a3])
guishear['pa']['value'] = np.array(dic['extShear PA'][a1,a2,a3])

for name in S.keys():
    s = S[name]
    if name == 'Source 1':
        for key in 'q','re','n','pa':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
        for key in 'x','y':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3] + dic['Lens 1 '+key][a1,a2,a3]) 
    elif name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
        for key in 'x','y':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3] + dic['Lens 1 '+key][a1,a2,a3])

print guishear
offsets[0][3] = np.array(dic['xoffset'][a1,a2,a3])
offsets[1][3] = np.array(dic['yoffset'][a1,a2,a3])
cPickle.dump([G,L,S,offsets,[guishear,True]],open('/data/ljo31/Lens/J1619/UpdatedGuiFile_'+str(X),'wb'))



