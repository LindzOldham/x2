import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles
from tools.simple import climshow


# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.99) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.99) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-50,vmax=50,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-75,vmax=75,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()



img = py.open('/data/ljo31b/lenses/J010127-334319_r_osj.fits')[0].data[460:-460,430:-490]/10.
bg=np.median(img[-10:,0:10])
img-=bg
sig = np.ones(img.shape)
sig[60:79,35:65] = 1e4
sig[10:30,0:15] = 1e4
psf = py.open('/data/ljo31b/lenses/r_psf1.fits')[0].data

psf /= psf.sum()
psf = convolve.convolve(img,psf)[1]

y,x = iT.overSample(img.shape,1)
rhs = (img/sig).ravel()
sflt = sig.ravel()
xflt = x.ravel()
yflt = y.ravel()

result = np.load('/data/ljo31b/lenses/model_twolens_4')
lp,trace,dic,_=result
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0

gal = SBObjects.Sersic('G1',{'x':dic['GX'][a1,a2,a3],'y':dic['GY'][a1,a2,a3],'re':dic['GR'][a1,a2,a3],'q':dic['GQ'][a1,a2,a3],'pa':dic['GP'][a1,a2,a3],'n':dic['GN'][a1,a2,a3]})
src = SBObjects.Sersic('S1',{'x':dic['SX'][a1,a2,a3],'y':dic['SY'][a1,a2,a3],'re':dic['SR'][a1,a2,a3],'q':dic['SQ'][a1,a2,a3],'pa':dic['SP'][a1,a2,a3],'n':dic['SN'][a1,a2,a3]})
src2 = SBObjects.Sersic('S2',{'x':dic['SX2'][a1,a2,a3],'y':dic['SY2'][a1,a2,a3],'re':dic['SR2'][a1,a2,a3],'q':1.,'pa':0.,'n':1.})

lens = MassModels.PowerLaw('L1',{'x':dic['GX'][a1,a2,a3],'y':dic['GY'][a1,a2,a3],'b':dic['LB'][a1,a2,a3],'q':dic['LQ'][a1,a2,a3],'pa':dic['LP'][a1,a2,a3],'eta':1.})
lens2 = MassModels.PowerLaw('L2',{'x':dic['SX'][a1,a2,a3],'y':dic['SY'][a1,a2,a3],'b':dic['LB2'][a1,a2,a3],'q':1.,'pa':0.,'eta':1.})
BETA = dic['beta'][a1,a2,a3]

def getModel(getResid=False):
    gal.setPars()
    src.setPars()
    src2.setPars()
    lens.setPars()
    lens2.setPars()

    #xl,yl = pylens.getDeflections([lens],[x,y])
    ax,ay = lens.deflections(x,y)

    lx = x-ax
    ly = y-ay

    ax2,ay2 = lens2.deflections(lx,ly)

    lx2 = x-BETA*ax-ax2
    ly2 = y-BETA*ay-ay2

    model = numpy.empty((sflt.size,4))
    model[:,0] = convolve.convolve(gal.pixeval(x,y),psf,False)[0].ravel()#/sflt
    model[:,1] = convolve.convolve(src.pixeval(lx,ly),psf,False)[0].ravel()#/sflt
    model[:,2] = convolve.convolve(src2.pixeval(lx2,ly2),psf,False)[0].ravel()#/sflt
    model[:,3] = np.ones(model[:,3].shape)

    fit,chi = optimize.nnls(model,rhs)
    if getResid is True:
        print fit
        fit[2] = 10.
        return (model.T*sflt).T*fit
    return chi


# now need to work out how to evaluate this and plot something...
def plotModel(getResid=False):
    gal.setPars()
    src.setPars()
    src2.setPars()
    lens.setPars()
    lens2.setPars()

    ax,ay = lens.deflections(x,y)

    lx = x-ax
    ly = y-ay

    ax2,ay2 = lens2.deflections(lx,ly)

    lx2 = x-BETA*ax-ax2
    ly2 = y-BETA*ay-ay2

    model = numpy.empty((sflt.size,4))
    model[:,0] = convolve.convolve(gal.pixeval(x,y),psf,False)[0].ravel()/sflt
    model[:,1] = convolve.convolve(src.pixeval(lx,ly),psf,False)[0].ravel()/sflt
    model[:,2] = convolve.convolve(src2.pixeval(lx2,ly2),psf,False)[0].ravel()/sflt
    model[:,3] = np.ones(model[:,3].shape)

    fit,chi = optimize.nnls(model,rhs)

    components = (model*fit).T.reshape((4,img.shape[0],img.shape[1]))
    model = components.sum(0)
    NotPlicely(img,model,sig)
    pl.show()
    for i in range(3):
        pl.figure()
        climshow(components[i])
        pl.colorbar()

plotModel()
pl.figure()
pl.plot(lp[:,0])
