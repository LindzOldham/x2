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

X = 29
print X
print 'eight cores only'

'''
X=0 - model3
X=1 - Vmodel5, Vband
X=2 - Vmodel5, Vband, bigger image. Testing to see what effect the use of a bigger image has, if any, by doing exactly the same procedure as X=1.
X=3 - 1_Iband. Put X=1 bak into gui and now on I band.
X=4 - 1_both4 - put X=1 onto both bands in the gui.
X = 5 - 1both5
X=6 - 2_Iband2. This is probably NEVER going to be a good fit.
X = 7 - 1both5_2src_2uniform TO RUN this will take forever TORUNTORUNTORUN (saved wrong)
X=8 - 7 again because calx079 doesn't seem to be doing much.
X=9 - gui_3c - put X=3 back into gui. 2/1/1, ovrs=2 (because I think we need it); upper limit on n_src increased to 10. TO RUN
X = 10 - 1both5 again, with better prior on n_src. So same as X=5, but with ovrs=2 and n_max = 10. RUNNING. Could fix the lens model, but I'm not sure that's a good idea yet? Yes - turning lens off anyway, as it speeds things up so much...
X=11 - 1_Iband again, lifting prior on n_src and lens_pa and having ovrs=2. Lenson. Ie. rerunning X=3.
X=12 - redoing 10 with lenson. Because things are moving around.
X=13 - gui_11_2src3. 2/1/2, with ovrs=2, I-band only!
X=14 - gui_11_2src3. 2/1/2, with ovrs=2, I-band only - restarting from the end of 13!
X = 15 - trying using flat noisemaps, 1both5 again (because why not?)
X=16 - model101. Started from scratch with a 2/1/1 model. Will probably be bad, but let's see...stick with flat noise maps.
X=17 - using psf2 instead of psf1 in both bands. 1both5 nochmal.
X=18 - model105, 3/1/1, ovrs=1 (for speed), V-band only
X=19 - model205, 2/1/1,ovrs=1 (for speed),V-band only
X=20 - model301 - 2/1/1,V-I,psd2big (though Matt reckons they're big enough anyway. Just in case...) ovrs=1.
X=21 - model306 - 2/1/1,psf4big (just trying different ones in case), ovrs=1 
X=22 - continuing X=17.
X=23 - gui_18_d. Putting X=18 onto V-I.
X=24 - gui_18_ib - putting X=18 on I.
X=25 doing X=24. X=24 was actually put onto the V-band.
X=26 - gui_25c. Putting X=25 onto V-I. 3/1/1
X=27 - gui_25_2srcb. Putting two sources in instead of one. 3/1/2. ovrs=1 (otherwise it will take forever!) For now, these sources are moving together. THis is a big problem with a huge parameter space.
X=28 - as x27, but now the sources are moving separately. ovrs=1,csub=11 ( better to have just two galaxies?)
X=29 - torun. LindzFit_J1619_nogui. Putting down objects in the terminal rather than in the gui. Slightly experimental...
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

img1 = py.open('/data/ljo31/Lens/J1619/F606W_sci_cutout_gross.fits')[0].data.copy()#[10:-10,10:-10]
sig1 = py.open('/data/ljo31/Lens/J1619/F606W_noisemap_gross.fits')[0].data.copy()#[10:-10,10:-10]
psf1 = py.open('/data/ljo31/Lens/J1619/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J1619/F814W_sci_cutout_gross.fits')[0].data.copy()#[10:-10,10:-10]
sig2 = py.open('/data/ljo31/Lens/J1619/F814W_noisemap_gross.fits')[0].data.copy()#[10:-10,10:-10]
psf2 = py.open('/data/ljo31/Lens/J1619/F814W_psf4big.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xo,xc=xo-45,xc-45
yo,yc=yo-50,yc-50
print np.mean(xo), np.mean(yo)
mask_V = py.open('/data/ljo31/Lens/J1619/maskgross.fits')[0].data.copy()#[10:-10,10:-10]
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

pars = []
cov = []
### first parameters need to be the offsets
pars.append(pymc.Uniform('xoffset',-5.,5.,value=1.7))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=0.7))
cov += [0.1,0.1]

# galaxies
gals = []
lenses = []
srcs = []
# galaxy 1
pars.append(pymc.Uniform('Galaxy 1 x',40.,50.,value=45.))
pars.append(pymc.Uniform('Galaxy 1 y',45.,55.,value=49.))
pars.append(pymc.Uniform('Galaxy 1 q',0.1,1.,value=0.6))
pars.append(pymc.Uniform('Galaxy 1 pa',-180.,180.,value=120.))
pars.append(pymc.Uniform('Galaxy 1 re',0.5,100.,value=3.))
pars.append(pymc.Uniform('Galaxy 1 n',0.5,8.,value=3.))
print '1'
gals.append(SBModels.Sersic('Galaxy 1',{'x':pars[-6],'y':pars[-5],'q':pars[-4],'pa':pars[-3],'re':pars[-2],'n':pars[-1]}))
cov += [0.1,0.1,0.1,50.,5.,0.5]
### galaxy 2
pars.append(pymc.Uniform('Galaxy 2 x',40.,50.,value=45.))
pars.append(pymc.Uniform('Galaxy 2 y',45.,55.,value=49.))
pars.append(pymc.Uniform('Galaxy 2 q',0.1,1.,value=0.6))
pars.append(pymc.Uniform('Galaxy 2 pa',-180.,180.,value=120.))
pars.append(pymc.Uniform('Galaxy 2 re',0.5,100.,value=20.))
pars.append(pymc.Uniform('Galaxy 2 n',0.5,8.,value=1.))
print '2'
gals.append(SBModels.Sersic('Galaxy 2',{'x':pars[-6],'y':pars[-5],'q':pars[-4],'pa':pars[-3],'re':pars[-2],'n':pars[-1]}))
cov += [0.1,0.1,0.1,50.,5.,0.5]
# lens 1
pars.append(pymc.Uniform('Lens 1 x',40.,50.,value=45.))
pars.append(pymc.Uniform('Lens 1 y',45.,55.,value=49.))
pars.append(pymc.Uniform('Lens 1 q',0.1,1.,value=0.95))
pars.append(pymc.Uniform('Lens 1 pa',-180.,180.,value=0.))
pars.append(pymc.Uniform('Lens 1 b',5.,15.,value=10.))
pars.append(pymc.Uniform('Lens 1 eta',0.5,1.5,value=1.2))
print '3'
lenses.append(MassModels.PowerLaw('Lens 1',{'x':pars[-6],'y':pars[-5],'q':pars[-4],'pa':pars[-3],'b':pars[-2],'eta':pars[-1]}))
cov += [0.1,0.1,0.1,10.,0.1,0.1]
# lens 2
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=-33.))
pars.append(pymc.Uniform('extShear',-0.1,0.1,value=-0.04))
lenses.append(MassModels.ExtShear('Shear',{'x':lenses[0].pars['x'],'y':lenses[0].pars['y'],'pa':pars[-2],'b':pars[-1]}))
cov += [50.,0.1]
# source 1
pars.append(pymc.Uniform('Source 1 x',42.,52.,value=47.))
pars.append(pymc.Uniform('Source 1 y',47.,57.,value=52.))
pars.append(pymc.Uniform('Source 1 q',0.1,1.,value=0.6))
pars.append(pymc.Uniform('Source 1 pa',-180.,180.,value=100.))
pars.append(pymc.Uniform('Source 1 re',0.5,100.,value=10.))
pars.append(pymc.Uniform('Source 1 n',0.5,10.,value=1.))
srcs.append(SBModels.Sersic('Source 1',{'x':pars[-6],'y':pars[-5],'q':pars[-4],'pa':pars[-3],'re':pars[-2],'n':pars[-1]}))
cov += [0.1,0.1,0.1,50.,1.,0.5]
# source 2
pars.append(pymc.Uniform('Source 2 x',42.,52.,value=47.))
pars.append(pymc.Uniform('Source 2 y',47.,57.,value=52.))
pars.append(pymc.Uniform('Source 2 q',0.1,1.,value=0.6))
pars.append(pymc.Uniform('Source 2 pa',-180.,180.,value=100.))
pars.append(pymc.Uniform('Source 2 re',0.5,100.,value=3.))
pars.append(pymc.Uniform('Source 2 n',0.5,10.,value=3.))
srcs.append(SBModels.Sersic('Source 2',{'x':pars[-6],'y':pars[-5],'q':pars[-4],'pa':pars[-3],'re':pars[-2],'n':pars[-1]}))
cov += [0.1,0.1,0.1,50.,1.,0.5]

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
        mask = masks[i]
        mask2 = mask2s[i]
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
        for src in srcs:
            src.setPars()
            tmp = xc*0.
            tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=11)
            tmp = iT.resamp(tmp,OVRS,True)
            tmp = convolve.convolve(tmp,psf,False)[0]
            model[n] = tmp[mask].ravel()
            n +=1
        model[n] = np.ones(model[n-1].shape)
        n+=1
        rhs = (imin/sigin) # data
        op = (model/sigin).T # model matrix
        fit, chi = optimize.nnls(op,rhs)
        model = (model.T*fit).sum(1)
        resid = (model-imin)/sigin
        lp += -0.5*(resid**2.).sum()
        models.append(model)
    return lp
 
  
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
S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=8,nwalkers=80,ntemps=4)
S.sample(1000)
print time.time()-start

outFile = '/data/ljo31/Lens/J1619/emcee'+str(X)
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
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=8,nwalkers=80,ntemps=4,initialPars=trace[-1])
    S.sample(1000)

    outFile = '/data/ljo31/Lens/J1619/emcee'+str(X)
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


# now showing a fit using the MASKED image
colours = ['F555W', 'F814W']
pl.figure()
for i in range(4):
    pl.plot(lp[:,i,:])
pl.show()

mask,mask2 = np.ones(img1.shape), np.ones(img1.shape)
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
        tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=11)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    model[n] = np.ones(model[n-1].shape)
    n+=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
    
