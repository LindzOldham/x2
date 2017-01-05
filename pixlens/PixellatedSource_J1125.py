import numpy,pyfits,pylab
import indexTricks as iT
from pylens import MassModels,pylens,adaptTools as aT,pixellatedTools as pT
from imageSim import SBModels,convolve
from scipy.sparse import diags
import pymc,cPickle
from scipy import optimize
import myEmcee_blobs as myEmcee #updateEmcee as myEmcee
import numpy as np, pylab as pl, pyfits as py
from pylens import lensModel
from scipy.interpolate import RectBivariateSpline
import adaptToolsBug as BB

img1 = py.open('/data/ljo31/Lens/J1125/F606W_sci_cutout_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50][30:-30,30:-30]
sig1 = py.open('/data/ljo31/Lens/J1125/F606W_noisemap_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50][30:-30,30:-30]
psf1 = py.open('/data/ljo31/Lens/J1125/F606W_psf3_filledin.fits')[0].data.copy()
psf1 = psf1[5:-7,5:-6]
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J1125/F814W_sci_cutout_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50][30:-30,30:-30]
sig2 = py.open('/data/ljo31/Lens/J1125/F814W_noisemap_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50][30:-30,30:-30]
psf2 = py.open('/data/ljo31/Lens/J1125/F814W_psf3_filledin.fits')[0].data.copy()
psf2 = psf2[5:-8,5:-6]
psf2 = psf2/np.sum(psf2)
Dx,Dy = -85,-85
OVRS=2
mask =  py.open('/data/ljo31/Lens/J1125/mask_huge2.fits')[0].data[50:-50,50:-50][50:-50,50:-50][30:-30,30:-30]


result = np.load('/data/ljo31/Lens/LensModels/twoband/J1125_212')
lp,trace,dic,_= result
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,xo=xc+Dx+80,xo+Dx+80
yc,yo=yc+Dy+80,yo+Dy+80
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2.T
mask2 = mask2==0
mask = mask==0

gals = []
for name in ['Galaxy 1','Galaxy 2']:
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
lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = dic['extShear'][a1,a2,a3] 
p['pa'] = dic['extShear PA'][a1,a2,a3]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in ['Source 2','Source 1']:
    p = {}
    if name == 'Source 2':
        for key in 'q','re','n','pa':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': 
            p[key] = dic[name+' '+key][a1,a2,a3] + lenses[0].pars[key]
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key] = srcs[0].pars[key] 
    srcs.append(SBModels.Sersic(name,p))

galsubs = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = dic['xoffset'][a1,a2,a3]
        dy = dic['yoffset'][a1,a2,a3]
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
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=31) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp = src.pixeval(x0,y0,1./OVRS,csub=31)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n]=np.ones(model[n].size)
    n+=1
    mmodel = model.reshape((n,image.shape[0],image.shape[1]))
    mmmodel = np.empty(((len(gals) + len(srcs)+1),image[mask].size))
    for m in range(mmodel.shape[0]):
        mmmodel[m] = mmodel[m][mask]
    op = (mmmodel/sigma[mask]).T
    rhs = image[mask]/sigma[mask]
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    n = 0
    galim = image.copy()
    for n in range(len(gals)):
        galim -=components[n]
    galsubs.append(galim)
    py.writeto('/data/ljo31/Lens/J1125/galsub_small_'+str(i)+'.fits',galim,clobber=True)

# we want the F606W image
galsub = galsubs[0]
pl.figure()
pl.imshow(galsubs[0],vmin=0,vmax=0.5,interpolation='nearest',origin='lower')
pl.colorbar()
pl.figure()
pl.imshow(galsubs[1],vmin=0,vmax=2,interpolation='nearest',origin='lower')
pl.colorbar()
pl.show()
# and a mask
mask = py.open('/data/ljo31/Lens/J1125/masknew2.fits')[0].data.copy()[30:-30,30:-30]
mask = mask==1
Npnts = 1  # Defines `fineness' of source reconstruction (bigger is coarser)

# Function to make a `nice' plot
def showRes(x,y,src,psf,img,sig,mask,iflt,vflt,cmat,reg,niter,npix):
    oy,ox = iT.coords((npix,npix))
    oy -= oy.mean()
    ox -= ox.mean()
    span = max(x.max()-x.min(),y.max()-y.min())
    oy *= span/npix
    ox *= span/npix
    ox += x.mean()
    oy += y.mean()
    lmat = psf*src.lmat
    rmat = src.rmat
    print reg
    res,fit,model,rhs,regg = aT.getModelG(iflt,vflt,lmat,cmat,rmat,reg,niter=niter)

    osrc = src.eval(ox.ravel(),oy.ravel(),fit).reshape(ox.shape)

    oimg = img*numpy.nan
    oimg[mask] = (lmat*fit)

    ext = [0,img.shape[1],0,img.shape[0]]
    ext2 = [x.mean()-span/2.,x.mean()+span/2.,y.mean()-span/2.,y.mean()+span/2.]
    pylab.figure()
    pylab.subplot(221)
    img[~mask] = numpy.nan
    pylab.imshow(img,origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=1,cmap='jet',aspect='auto')
    pylab.colorbar()
    pylab.subplot(222)
    pylab.imshow(oimg,origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=1,cmap='jet',aspect='auto')
    pylab.colorbar()
    pylab.subplot(223)
    pylab.imshow((img-oimg)/sig,origin='lower',interpolation='nearest',extent=ext,vmin=-3,vmax=3,cmap='jet',aspect='auto')
    pylab.colorbar()
    pylab.subplot(224)
    pylab.imshow(osrc,origin='lower',interpolation='nearest',extent=ext2,vmin=0,vmax=1,cmap='jet',aspect='auto')
    pylab.colorbar()
    return osrc

# Setup data for adaptive source modelling
img,sig,psf = galsub, sig1,psf1
pl.figure()
pl.imshow(img,vmin=0,vmax=1,origin='lower')
pl.colorbar()
y,x = iT.coords(img.shape)
x=x+Dx+80
y=y+Dy+80
#x,y=x+dx,y+dy
cpsf = convolve.convolve(img,psf)[1]
ifltm = img[mask]
sfltm = sig[mask]
vfltm = sfltm**2
cmatm = diags(1./sfltm,0)
xm = x[mask]
ym = y[mask]
coords = [xm,ym]

PSF = pT.getPSFMatrix(psf,img.shape)
PSFm = pT.maskPSFMatrix(PSF,mask)

iflt = img.flatten()
sflt = sig.flatten()
vflt = sflt**2

xflt = x.flatten()
yflt = y.flatten()

src = aT.AdaptiveSource(ifltm/sfltm,ifltm.size/Npnts)

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
# casting source pixels back and regularising
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,1e2,1,100)
pylab.show()

los = dict([('q',0.05),('pa',-180.),('re',0.1),('n',0.5)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',10.)])
covs = dict([('x',0.05),('y',0.05),('q',0.1),('pa',1.),('re',0.5),('n',0.5)])
covlens = dict([('x',0.05),('y',0.05),('q',0.05),('pa',1.),('b',0.2),('eta',0.1)])
lenslos, lenshis = dict([('q',0.05),('pa',-180.),('b',0.5),('eta',0.5)]), dict([('q',1.00),('pa',180.),('b',100.),('eta',1.5)])

result = np.load('/data/ljo31/Lens/J1125/pixsrc_3')
lp,trace,dic,_= result
a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2 = 0


pars = []
cov = []
lenses = []
lpars = {}
for key in 'x','y','q','pa','b','eta':
    val = dic['Lens 1 '+key][a1,a2,a3]
    if key == 'x' or key == 'y':
        lo,hi=val-5.,val+5.
    else:
        lo,hi = lenslos[key],lenshis[key]
    pars.append(pymc.Uniform('Lens 1 '+key,lo,hi,value=val))
    lpars[key] = pars[-1]
    cov.append(covlens[key])
lenses.append(MassModels.PowerLaw('Lens 1',lpars))

spars = {}
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,a2,a3]))
pars.append(pymc.Uniform('extShear PA',-180.,180,value=dic['extShear PA'][a1,a2,a3]))
spars['b']=pars[-2]
spars['pa']=pars[-1]
spars['x'] = lpars['x']
spars['y'] = lpars['y']
cov += [0.05,0.5]
lenses.append(MassModels.ExtShear('shear',spars))

reg=1.
previousResult = None

import time
def doFit(p=None,doReg=True,updateReg=True,checkImgs=True,levMar=False):
    global reg
    # Check if using levMar-style parameters
    if p is not None:
        for i in range(len(p)):
            pars[i].value = p[i]
            # If the parameter is out-of-bounds return a bad fit
            try:
                a = pars[i].logp
            except:
                return iflt/sflt

    for l in lenses:
        l.setPars()
    xl,yl = pylens.getDeflections(lenses,coords)

    src.update(xl,yl,doReg=doReg)
    lmat = PSFm*src.lmat
    if doReg==True:
        rmat = src.rmat
    else:
        rmat = None
    nupdate = 0
    if doReg==True and updateReg==True:
        nupdate = 10
    res,fit,model,_,regg = aT.getModelG(ifltm,vfltm,lmat,cmatm,rmat,reg,nupdate)
    reg = regg[0]
    if checkImgs is False:
        if levMar:
            res = res**0.5+ifltm*0.
        return -0.5*res
    # This checks if images are formed outside of the masked region
    xl,yl = pylens.getDeflections(lenses,[xflt,yflt])
    oimg,pix = src.eval(xl,yl,fit,domask=False)
    oimg = PSF*oimg
    res = (iflt-oimg)/sflt
    if levMar:
        return res
    return -0.5*(res**2).sum()


@pymc.observed
def likelihood(value=0.,tmp=pars):
    return doFit(None,True,True,True,False)

cov = numpy.array(cov)
if previousResult is not None:
    result = numpy.load(previousResult)
    lp = result[0]
    trace = numpy.array(result[1])
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,i]
    ns,nw,np = trace.shape
    cov = numpy.cov(trace[ns/2:].reshape((ns*nw/2,np)).T)

print 'about to do doFit - i.e. get the regularisation for the current model'

doFit(None,True,True,False)
doFit(None,True,True,False)

print 'done doFit'

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,10,400)
pylab.show()

S = myEmcee.PTEmcee(pars+[likelihood],cov=cov,nthreads=32,nwalkers=80,ntemps=3)
S.sample(500)

print 'done emcee'
outFile = '/data/ljo31/Lens/J1125/pixsrc_4'

f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

print 'cPickled!'

result = S.result()
lp,trace,dic,_ = result
a1,a2 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,0,a2,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

print reg
doFit(None,True,True,False)
doFit(None,True,True,False)
print reg
xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.savefig('EIGHT.png')

jj=0
for jj in range(30):
    S.p0 = trace[-1]
    print 'sampling'
    S.sample(500)

    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()

    result = S.result()
    lp = result[0]

    trace = numpy.array(result[1])
    a1,a2 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,0,a2,i]
    print jj
    jj+=1
    doFit(None,True,True,False)
    doFit(None,True,True,False)
    print reg
    xl,yl = pylens.getDeflections(lenses,coords)
    src.update(xl,yl)
    osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
    pylab.savefig('EIGHT_iterated.png')

print 'das Ende'

print reg
doFit(None,True,True,False)
doFit(None,True,True,False)
print reg
xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.savefig('EIGHT.png')
pylab.show()
