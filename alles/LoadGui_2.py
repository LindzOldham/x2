import numpy as np, pylab as pl, pyfits as py
import cPickle,pymc
from pylens import MassModels,pylens,adaptTools as aT,pixellatedTools as pT
import indexTricks as iT
from scipy.sparse import diags
import updateEmcee as myEmcee
from scipy import optimize


imgName = '/data/ljo31/Lens/SDSSJ1606+2235_F606W_sci_cutout.fits'
sigName = '/data/ljo31/Lens/SDSSJ1606+2235_F606W_noise3_cutout.fits'
psfName = '/data/ljo31/Lens/SDSSJ1606+2235_F606W_psf.fits'

img2Name = '/data/ljo31/Lens/SDSSJ1606+2235_F814W_sci_cutout.fits'
sig2Name = '/data/ljo31/Lens/SDSSJ1606+2235_F814W_noise3_cutout.fits'
psf2Name = '/data/ljo31/Lens/SDSSJ1606+2235_F814W_psf.fits'

guiFile = '/data/ljo31/Lens/ModelFit7'
outFile = '/data/ljo31/Lens/result.dat'

previousResult = None
nsamples = 500                     # Number of samples per walker
nwalkers = 20                      # Number of walkers
nthreads = 1                        # Number of CPU cores
previousResult = None               # Filename of previous run



img1 = py.open(imgName)[0].data.copy()
sig1 = py.open(sigName)[0].data.copy()
psf1 = py.open(psfName)[0].data.copy()

img2 = py.open(img2Name)[0].data.copy()
sig2 = py.open(sig2Name)[0].data.copy()
psf2 = py.open(psf2Name)[0].data.copy()

blue = [img1,sig1,psf1]
red = [img2,sig2,psf2]

# create a colour image
import colorImage
CI = colorImage.ColorImage()
img3 = (img1 + img2)/2.

img = CI.createModel(img1,img3,img2)
#pl.figure()
#pl.imshow(img,origin='lower',interpolation='nearest')
#pl.gca().xaxis.set_ticks([])
#pl.gca().yaxis.set_ticks([])

# load up the GUI model
G,L,S,offsets,shear = np.load(guiFile) # galaxy, lens, source
pars = []
cov = []
lenses = []
for name in L.keys():
    model = L[name]
    lpars = {}
    for key in 'x','y','q','pa','b','eta':
        if model[key]['type']=='constant': # if it had been fixed
            lpars[key] = model[key]['value']
        else: # ie. either uniform or normal in the gui
            lo,hi,val = model[key]['lower'],model[key]['upper'],model[key]['value']
            pars.append(pymc.Uniform('%s:%s'%(name,key),lo,hi,value=val))
            lpars[key] = pars[-1]
            cov.append(model[key]['sdev'])
    lenses.append(MassModels.PowerLaw(name,lpars))

y,x = iT.coords(img[:,:,0].shape)
coords = [x,y]
for l in lenses:
    l.setPars()
xl,yl = pylens.getDeflections(lenses,coords)
#pl.figure()
#pl.scatter(xl,yl,c=img[:,:,0],s=100)
#pl.figure()
#pl.scatter(x,y,c=img[:,:,0],s=100)

# pylens.getDeflections calculates alpha using the massmodel.deflections routine, then subtracts it from the original coordinates (image plane coordinates) to give source plane coordinates.

# now we have to somehow set up all the modelly stuff.


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
    res,fit,model,rhs,regg = aT.getModelG(iflt,vflt,lmat,cmat,rmat,reg,niter=niter)

    osrc = src.eval(ox.ravel(),oy.ravel(),fit).reshape(ox.shape)

    oimg = img*np.nan
    oimg[mask] = (lmat*fit)

    ext = [0,img.shape[1],0,img.shape[0]]
    ext2 = [x.mean()-span/2.,x.mean()+span/2.,y.mean()-span/2.,y.mean()+span/2.]
    pl.figure()
    pl.subplot(221)
    img[~mask] = np.nan
    pl.imshow(img,origin='lower',interpolation='nearest',extent=ext)
    pl.colorbar()
    pl.title('original image')
    pl.subplot(222)
    pl.imshow(oimg,origin='lower',interpolation='nearest',extent=ext)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow((img-oimg)/sig,origin='lower',interpolation='nearest',extent=ext)
    pl.colorbar()
    pl.title('residuals')
    pl.subplot(224)
    pl.imshow(osrc,origin='lower',interpolation='nearest',extent=ext2)
    pl.colorbar()
    pl.title('source plane?')
    return osrc

# for now, we'll just use a single colour of lens
img = img[:,:,0]
mask = img*0
mask = np.where(mask==0,True,False)
# nicht sicher

ifltm = img1[mask]
sfltm = sig1[mask]
vfltm = sfltm**2 # variance from standard deviation.
cmatm = diags(1./sfltm,0) # make a diagnonal covariance matrix
xm = x[mask]
ym = y[mask]
coords = [xm,ym]

PSF = pT.getPSFMatrix(psf1,img1.shape) # 2750x2750 - why so large?
PSFm = pT.maskPSFMatrix(PSF,mask)


# for us, as we have no mask, this is a repeat of the above
iflt = img1.flatten()
sflt = sig1.flatten()
vflt = sflt**2

xflt = x.flatten()
yflt = y.flatten()

Npnts = 4  # Defines `fineness' of source reconstruction (bigger is coarser)
src = aT.AdaptiveSource(ifltm/sfltm,ifltm.size/Npnts)
reg = 1.

# do fit?
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
    oimg,pix = src.eval(xl,yl,fit,domask=False) # evaluate new model in the image plane (using the source plane coordinates to evaluate the model)
    oimg = PSF*oimg
    res = (iflt-oimg)/sflt # new residuals
    if levMar:
        return res
    return -0.5*(res**2).sum() # likelihood


@pymc.observed
def likelihood(value=0.,tmp=pars):
    return doFit(None,True,False,True,False) #  doing regularisation; not updating regularisation; checking image; not levMar

cov = np.array(cov)
if previousResult is not None:
    result = np.load(previousResult)
    lp = result[0]
    trace = np.array(result[1])
    a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,i]
    ns,nw,np = trace.shape
    cov = np.cov(trace[ns/2:].reshape((ns*nw/2,np)).T)

doFit(None,True,True,False) # doing regularisation, updating regularisation, that is all.
doFit(None,True,True,False)


reg *= 10.
S = myEmcee.Emcee(pars+[likelihood],cov=cov,nthreads=nthreads,nwalkers=nwalkers)
S.sample(nsamples)

f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

result = S.result()
lp = result[0]
trace = np.array(result[1])
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img1,sig1,mask,ifltm,vfltm,cmatm,reg,0,400)

# is the idea to mask the galaxy?
