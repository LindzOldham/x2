import numpy,pyfits,pylab
import indexTricks as iT
from pylens import MassModels,pylens,adaptTools as aT,pixellatedTools as pT
from scipy.sparse import diags
import pymc,cPickle
from scipy import optimize
import updateEmcee as myEmcee
import numpy as np, pylab as pl

nsamples = 500                     # Number of samples per walker
nwalkers = 50                      # Number of walkers
nthreads = 4                        # Number of CPU cores
previousResult = None               # Filename of previous run


img1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2_masked.fits')[0].data.copy() # masking out the possiblyasecondsource regions!
psf1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)

img = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf= psf[15:-16,14:-16]
psf /= psf.sum()

guiFile = '/data/ljo31/Lens/J1605/terminal_iterated_4'
outFile = '/data/ljo31/Lens/J1605/pixsrc'

mask = pyfits.open('/data/ljo31/Lens/J1605/mask.fits')[0].data.copy()
mask = mask==1

Npnts = 4  # Defines `fineness' of source reconstruction (bigger is coarser)

# Read in model from gui save file and create variables and models
G,L,S,offsets,shear = numpy.load(guiFile)
pars = []
cov = []
lenses = []
for name in L.keys():
    model = L[name]
    lpars = {}
    for key in 'x','y','q','pa','b','eta':
        if model[key]['type']=='constant':
            lpars[key] = model[key]['value']
        else:
            lo,hi,val = model[key]['lower'],model[key]['upper'],model[key]['value']
            pars.append(pymc.Uniform('%s:%s'%(name,key),lo,hi,value=val))
            lpars[key] = pars[-1]
            cov.append(model[key]['sdev'])
    lenses.append(MassModels.PowerLaw(name,lpars))

if shear[1]==1:
    model = shear[0]
    spars = {}
    for key in 'x','y','b','pa':
        if model[key]['type']=='constant':
            spars[key] = model[key]['value']
        else:
            lo,hi,val = model[key]['lower'],model[key]['upper'],model[key]['value']
            pars.append(pymc.Uniform('shear:%s'%(key),lo,hi,value=val))
            spars[key] = pars[-1]
            cov.append(model[key]['sdev'])
    lenses.append(MassModels.ExtShear('shear',spars))


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

    oimg = img*numpy.nan
    oimg[mask] = (lmat*fit)

    ext = [0,img.shape[1],0,img.shape[0]]
    ext2 = [x.mean()-span/2.,x.mean()+span/2.,y.mean()-span/2.,y.mean()+span/2.]
    pylab.figure()
    pylab.subplot(221)
    img[~mask] = numpy.nan
    pylab.imshow(img,origin='lower',interpolation='nearest',extent=ext)
    pylab.colorbar()
    pylab.subplot(222)
    pylab.imshow(oimg,origin='lower',interpolation='nearest',extent=ext)
    pylab.colorbar()
    pylab.subplot(223)
    pylab.imshow((img-oimg)/sig,origin='lower',interpolation='nearest',extent=ext)
    pylab.colorbar()
    pylab.subplot(224)
    pylab.imshow(osrc,origin='lower',interpolation='nearest',extent=ext2)
    pylab.colorbar()
    return osrc


# Setup data for adaptive source modelling
y,x = iT.coords(img.shape)

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
reg = 1.

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
    # This checks is images are formed outside of the masked region
    xl,yl = pylens.getDeflections(lenses,[xflt,yflt])
    oimg,pix = src.eval(xl,yl,fit,domask=False)
    oimg = PSF*oimg
    res = (iflt-oimg)/sflt
    if levMar:
        return res
    return -0.5*(res**2).sum()


@pymc.observed
def likelihood(value=0.,tmp=pars):
    return doFit(None,True,False,True,False)

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

print 'about to do doFit'

doFit(None,True,True,False)
doFit(None,True,True,False)

print 'done doFit'

#xl,yl = pylens.getDeflections(lenses,coords)
#src.update(xl,yl)
#osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,10,400)
#pylab.show()

#start = [i.value for i in pars]
#result = optimize.leastsq(doFit,start,args=(True,True,True,True))[0]
#for i in range(len(pars)):
#    pars[i].value = result[i]
#print start
#print result

#xl,yl = pylens.getDeflections(lenses,coords)
#src.update(xl,yl)
#osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,10,400)
#pylab.show()

reg *= 10.
S = myEmcee.Emcee(pars+[likelihood],cov=cov,nthreads=nthreads,nwalkers=50)
S.sample(300)

print 'done emcee'

f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

print 'cPickled!'

result = S.result()
lp = result[0]
trace = numpy.array(result[1])
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

print 'should have printed parametrers now...!'

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.show()

print 'das Ende'
