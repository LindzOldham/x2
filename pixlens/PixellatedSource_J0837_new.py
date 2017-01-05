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

from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'onut'


img1 = py.open('/data/ljo31/Lens/J0837/galsub_0.fits')[0].data.copy()[80:-80,80:-80][40:-50,40:-40]
sig1 = py.open('/data/ljo31/Lens/J0837/F606W_noisemap_huge.fits')[0].data.copy()[80:-80,80:-80][40:-50,40:-40] 
psf1 = py.open('/data/ljo31/Lens/J0837/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J0837/galsub_1.fits')[0].data.copy()[80:-80,80:-80][40:-50,40:-40]
sig2 = py.open('/data/ljo31/Lens/J0837/F814W_noisemap_huge.fits')[0].data.copy()[80:-80,80:-80][40:-50,40:-40]
psf2 = py.open('/data/ljo31/Lens/J0837/F814W_psf3.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)
Dx,Dy = -100+40,-100.+40

result = np.load('/data/ljo31/Lens/LensModels/twoband/J0837_211')
#result = np.load('/data/ljo31/Lens/J0837/pixsrc_neu3')

lp,trace,dic,_= result
a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2 = 0

# and a mask
mask = py.open('/data/ljo31/Lens/J0837/mask22.fits')[0].data.copy()[80:-80,80:-80][40:-50,40:-40]
mask = mask==1
Npnts = 2  # Defines `fineness' of source reconstruction (bigger is coarser)

vx,vy = dic['Lens 1 x'][a1,0,a3], dic['Lens 1 y'][a1,0,a3]
X = pymc.Uniform('Lens 1 x',vx-5,vx+5,value=vx)
Y = pymc.Uniform('Lens 1 y',vy-5,vy+5,vy)
B = pymc.Uniform('Lens 1 b',0.5,100.,value=dic['Lens 1 b'][a1,0,a3])
Q = pymc.Uniform('Lens 1 q',0.1,1.0,value=dic['Lens 1 q'][a1,0,a3])
ETA = pymc.Uniform('Lens 1 eta',0.5,1.5,value=dic['Lens 1 eta'][a1,0,a3])
PA = pymc.Uniform('Lens 1 pa',-180,180.,value=dic['Lens 1 pa'][a1,0,a3])

SH = pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,0,a3])
SHPA = pymc.Uniform('extShear PA',-180.,0,value=dic['extShear PA'][a1,0,a3])

lens1 = MassModels.PowerLaw('Lens 1',{'x':X,'y':Y,'b':B,'eta':ETA,'q':Q,'pa':PA})
shear = MassModels.ExtShear('shear',{'x':X,'y':Y,'b':SH,'pa':SHPA})
lenses = [lens1,shear]
pars = [X,Y,B,Q,ETA,PA,SH,SHPA]
cov = [0.1,0.1,0.1,0.1,0.1,1.,0.05,10.]
#pars = [X,Y,Q,ETA,PA,SH,SHPA]
#cov = [0.1,0.1,0.1,0.1,1.,0.05,10.]
print 'b', B

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
    print regg
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

    # for paper -- just model in the two planes?
    '''pylab.figure(figsize=(12,5))
    pylab.subplot(121)
    pl.figtext(0.05,0.8,'J0837')
    pylab.imshow(oimg,origin='lower',interpolation='nearest',extent=ext,vmin=0,vmax=1,cmap='jet',aspect='auto')
    #ax = pl.axes()
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    pl.gca().xaxis.set_major_locator(pl.NullLocator())
    pl.gca().yaxis.set_major_locator(pl.NullLocator())
    pylab.colorbar()
    pylab.subplot(122)
    pylab.imshow(osrc[:-10,:-55],origin='lower',interpolation='nearest',extent=ext2,vmin=0,vmax=1,cmap='jet',aspect='auto')
    pl.gca().xaxis.set_major_locator(pl.NullLocator())
    pl.gca().yaxis.set_major_locator(pl.NullLocator())
    pylab.colorbar()'''
    return osrc

# Setup data for adaptive source modelling
img,sig,psf = img1, sig1,psf1
y,x = iT.coords(img.shape)
x=x+Dx+80
y=y+Dy+80
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
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,4.,0,100)
pylab.show()


reg=4.
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

print 'about to do doFit - i.e. get the regularisation for the current model'

doFit(None,True,False,False)
doFit(None,True,False,False)
print 'done doFit'

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
print 'reg',reg
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.show()
print 'reg',reg
reg=4.
print 'reg',reg

S = myEmcee.PTEmcee(pars+[likelihood],cov=cov,nthreads=24,nwalkers=100,ntemps=3)
S.sample(500)

print 'done emcee'
outFile = '/data/ljo31/Lens/J0837/pixsrc_neu4_optreg'

f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

print 'cPickled!'

result = S.result()
lp,trace,dic,_ = result
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
for i in range(len(pars)):
    pars[i].value = trace[a1,0,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

print 'reg',reg
doFit(None,True,False,False)
doFit(None,True,False,False)
print 'reg',reg
xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
reg=4.

jj=0
for jj in range(12):
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
        pars[i].value = trace[a1,0,a3,i]
    print jj
    jj+=1
    doFit(None,True,False,False)
    doFit(None,True,False,False)
    print reg
    xl,yl = pylens.getDeflections(lenses,coords)
    src.update(xl,yl)
    reg = 4.

print 'das Ende'

print reg
doFit(None,True,False,False)
doFit(None,True,False,False)
print reg
xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.show()

