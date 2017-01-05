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

img1 = py.open('/data/ljo31/Lens/J1125/galsub_0.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1125/F606W_noisemap_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50]
psf1 = py.open('/data/ljo31/Lens/J1125/F606W_psf3_filledin.fits')[0].data.copy()
psf1 = psf1[5:-7,5:-6]
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J1125/galsub_1.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1125/F814W_noisemap_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50]
psf2 = py.open('/data/ljo31/Lens/J1125/F814W_psf3_filledin.fits')[0].data.copy()
psf2 = psf2[5:-8,5:-6]
psf2 = psf2/np.sum(psf2)
Dx,Dy = -85,-85

result = np.load('/data/ljo31/Lens/LensModels/twoband/J1125_212')
lp,trace,dic,_= result
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

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

# and a mask
mask = py.open('/data/ljo31/Lens/J1125/maskneu.fits')[0].data.copy()#[50:-50,50:-50]
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
    pylab.imshow(img[50:-50,46:-50],origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=1,cmap='jet',aspect='auto')
    pylab.colorbar()
    pylab.subplot(222)
    pylab.imshow(oimg[50:-50,46:-50],origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=1,cmap='jet',aspect='auto')
    pylab.colorbar()
    pylab.subplot(223)
    pylab.imshow((img[50:-50,46:-50]-oimg[50:-50,46:-50])/sig[50:-50,46:-50],origin='lower',interpolation='nearest',extent=ext,vmin=-3,vmax=3,cmap='jet',aspect='auto')
    pylab.colorbar()
    pylab.subplot(224)
    pylab.imshow(osrc,origin='lower',interpolation='nearest',extent=ext2,vmin=0,vmax=2,cmap='jet',aspect='auto')
    pylab.colorbar()

    # for paper -- just model in the two planes
    pylab.figure(figsize=(12,5))
    pylab.subplot(121)
    pl.figtext(0.05,0.8,'J1125',fontsize=30)
    pylab.imshow(oimg[50:-50,46:-50],origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=1,cmap='jet',aspect='auto')
    #ax = pl.axes()
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    pl.gca().xaxis.set_major_locator(pl.NullLocator())
    pl.gca().yaxis.set_major_locator(pl.NullLocator())
    pylab.colorbar()
    pylab.subplot(122)
    pylab.imshow(osrc[:],origin='lower',interpolation='nearest',extent=ext2,vmin=0,vmax=2,cmap='jet',aspect='auto')
    pl.gca().xaxis.set_major_locator(pl.NullLocator())
    pl.gca().yaxis.set_major_locator(pl.NullLocator())
    pylab.colorbar()

    return osrc

# Setup data for adaptive source modelling
img,sig,psf = img1, sig1,psf1
y,x = iT.coords(img.shape)
x=x+Dx+50
y=y+Dy+50
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

los = dict([('q',0.05),('pa',-180.),('re',0.1),('n',0.5)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',10.)])
covs = dict([('x',0.05),('y',0.05),('q',0.1),('pa',1.),('re',0.5),('n',0.5)])
covlens = dict([('x',0.05),('y',0.05),('q',0.05),('pa',1.),('b',0.2),('eta',0.1)])
lenslos, lenshis = dict([('q',0.05),('pa',-180.),('b',0.5),('eta',0.5)]), dict([('q',1.00),('pa',180.),('b',100.),('eta',1.5)])

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
print reg
print 'done doFit'

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,10,400)
pylab.show()
