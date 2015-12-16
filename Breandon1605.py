import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
import pylab as pl
import numpy as np
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline

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

lens = np.loadtxt('/data/ljo31/Lens/J1605/BrendonLens2.txt')
source = np.loadtxt('/data/ljo31/Lens/J1605/BrendonSource.txt')
mediansl = np.percentile(lens, 50,axis=0)
medianss = np.percentile(source, 50,axis=0)
Ie,m,re,xcs,ycs,qs,thetas = medianss
b,ql,xcl,ycl,thetal,shearl, thetashearl = mediansl


img1 = py.open('/data/ljo31/public_html/Lens/J1605/galsub_F555W.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2.fits')[0].data.copy() # masking out the possiblyasecondsource regions!
psf1 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/public_html/Lens/J1605/galsub_F814W.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf2= psf2[15:-16,14:-16]
psf2 /= psf2.sum()

det = numpy.load('/data/ljo31/Lens/J1605/detFINAL.npy')[()]

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,yc,xo,yo = xc-45.5,yc-45.5,xo-45.5,yo-45.5

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

lenses = []
src = SBModels.Sersic('Source 1', {'x':xcs/0.05,'y':ycs/0.05,'q':qs,'re':re/0.05,'amp':Ie,'n':m,'pa':thetas*180/np.pi})
lenses.append(MassModels.PowerLaw('Lens 1', {'x':xcl/0.05,'y':ycl/0.05,'q':ql,'b':b/0.05,'pa':thetal*180/np.pi,'eta':1}))
lenses.append(MassModels.ExtShear('shear', {'x':xcl/0.05,'y':ycl/0.05,'b':shearl,'pa':thetashearl*180/np.pi}))

for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = det['xoffset'][-1]
        dy = det['yoffset'][-1]
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    print xp.shape, xin.shape
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,src,[xp,yp],1./OVRS,getPix=True)
    src.setPars()
    tmp = xc*0.
    tmp = src.pixeval(x0,y0,1./OVRS,csub=1)
    tmp = iT.resamp(tmp,OVRS,True)
    tmp = convolve.convolve(tmp,psf,False)[0]
    model = tmp.copy()
    NotPlicely(image,model,sigma)
    #pl.figure()
    #pl.imshow(model,origin='lower')
