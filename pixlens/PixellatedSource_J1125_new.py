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
from linslens.pixplot import *

img1 = py.open('/data/ljo31/Lens/J1125/galsub_0.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1125/F606W_noisemap_huge2.fits')[0].data.copy()[50:-50,50:-50][50:-50,50:-50]
psf1 = py.open('/data/ljo31/Lens/J1125/F606W_psf3_filledin.fits')[0].data.copy()
psf1 = psf1[5:-7,5:-6]
psf1 = psf1/np.sum(psf1)

Dx,Dy = -85,-85

#result = np.load('/data/ljo31/Lens/LensModels/twoband/J1125_212')
#lp,trace,dic,_= result
#a2=0.
#a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

result = np.load('/data/ljo31/Lens/J1125/pixsrc_neu2')
lp,trace,dic,_= result
a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)


vx,vy = dic['Lens 1 x'][a1,0,a3], dic['Lens 1 y'][a1,0,a3]
X = pymc.Uniform('Lens 1 x',vx-5,vx+5,vx)
Y = pymc.Uniform('Lens 1 y',vy-5,vy+5,vy)
B = pymc.Uniform('Lens 1 b',0.5,100.,value=dic['Lens 1 b'][a1,0,a3])
Q = pymc.Uniform('Lens 1 q',0.1,1.0,value=dic['Lens 1 q'][a1,0,a3])
ETA = pymc.Uniform('Lens 1 eta',0.5,1.5,value=dic['Lens 1 eta'][a1,0,a3])
PA = pymc.Uniform('Lens 1 pa',0,180.,value=dic['Lens 1 pa'][a1,0,a3])

SH = pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,0,a3])
SHPA = pymc.Uniform('extShear PA',-180.,180,value=dic['extShear PA'][a1,0,a3])

lens1 = MassModels.PowerLaw('Lens 1',{'x':X,'y':Y,'b':B,'eta':ETA,'q':Q,'pa':PA})
shear = MassModels.ExtShear('shear',{'x':X,'y':Y,'b':SH,'pa':SHPA})
lenses = [lens1,shear]
pars = [X,Y,B,Q,ETA,PA,SH,SHPA]
cov = [0.1,0.1,0.1,0.1,0.1,1.,0.05,10.]

# and a mask
mask = py.open('/data/ljo31/Lens/J1125/maskneu.fits')[0].data.copy()#[50:-50,50:-50]
mask = py.open('/data/ljo31/Lens/J1125/maskfin.fits')[0].data.copy()#[50:-50,50:-50]

mask = mask==1
Npnts = 2  # Defines `fineness' of source reconstruction (bigger is coarser)


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

reg=1.

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
print reg
print 'done doFit'

xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.show()

reg=1.
S = myEmcee.PTEmcee(pars+[likelihood],cov=cov,nthreads=24,nwalkers=100,ntemps=3)
S.sample(500)

print 'done emcee'
outFile = '/data/ljo31/Lens/J1125/pixsrc_neu4'

f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

print 'cPickled!'

result = S.result()
lp,trace,dic,_ = result
#a1,a2 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
#for i in range(len(pars)):
#    pars[i].value = trace[a1,0,a2,i]
#    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,0,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

print reg
doFit(None,True,False,False)
doFit(None,True,False,False)
print reg
xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.close('all')

jj=0
for jj in range(20):
    S.p0 = trace[-1]
    print 'sampling'
    S.sample(500)

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
    osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
    pylab.close('all')
    #reg=1.

print 'das Ende'

print reg
doFit(None,True,False,False)
doFit(None,True,False,False)
print reg
xl,yl = pylens.getDeflections(lenses,coords)
src.update(xl,yl)
osrc = showRes(xl,yl,src,PSFm,img,sig,mask,ifltm,vfltm,cmatm,reg,0,400)
pylab.show()

