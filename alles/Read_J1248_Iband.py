import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles


# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.5) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.5) #,vmin=vmin,vmax=vmax)
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
    py.writeto('/data/ljo31/Lens/J0837/resid.fits',(image-im),clobber=True)



img1 = py.open('/data/ljo31/Lens/J1248/F555W_sci_cutout.fits')[0].data.copy()[10:-10,20:-25]
sig1 = py.open('/data/ljo31/Lens/J1248/F555W_noisemap.fits')[0].data.copy()[10:-10,20:-25]
psf1 = py.open('/data/ljo31/Lens/J1248/F555W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1248/galsub_1.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1248/F814W_noisemap.fits')[0].data.copy()[10:-10,20:-25]
psf2 = py.open('/data/ljo31/Lens/J1248/F814W_psf1.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)

result = np.load('/data/ljo31/Lens/J1248/new1_Iband_5')
lp,trace,dic,_ = result
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

imgs = [img2]
sigs = [sig2]
psfs = [psf2]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xo,xc=xo+10,xc+10
mask = np.zeros(img1.shape)
mask = py.open('/data/ljo31/Lens/J1248/mask100.fits')[0].data.copy()#[35:-40,30:-25]
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(yc,xc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==1
mask = mask==1

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)


src1 = SBModels.Sersic('Source 1',{'x':dic['Source 1 x'][a1,0,a3],'y':dic['Source 1 y'][a1,0,a3],'n':dic['Source 1 n'][a1,0,a3],'re':dic['Source 1 re'][a1,0,a3],'pa':dic['Source 1 pa'][a1,0,a3],'q':dic['Source 1 q'][a1,0,a3]})
#src2 = SBModels.Sersic('Source 2',{'x':dic['Source 2 x'][a1,0,a3],'y':dic['Source 2 y'][a1,0,a3],'n':dic['Source 2 n'][a1,0,a3],'re':dic['Source 2 re'][a1,0,a3],'pa':dic['Source 2 pa'][a1,0,a3],'q':dic['Source 2 q'][a1,0,a3]})
srcs = [src1]#,src2]


lresult = np.load('/data/ljo31/Lens/J1248/pixsrc_2_ctd_new')
llp,ltrace,ldic,_ = lresult
la2=0.
la1,la3 = numpy.unravel_index(llp[:,0].argmax(),llp[:,0].shape)


lens1 = MassModels.PowerLaw('Lens 1',{'x':dic['Lens 1 x'][a1,0,a3],'y':dic['Lens 1 y'][a1,0,a3],'b':dic['Lens 1 b'][a1,0,a3],'eta':dic['Lens 1 eta'][a1,0,a3],'q':dic['Lens 1 q'][a1,0,a3],'pa':dic['Lens 1 pa'][a1,0,a3]})
shear = MassModels.ExtShear('shear',{'x':dic['Lens 1 x'][a1,0,a3],'y':dic['Lens 1 y'][a1,0,a3],'b':ldic['extShear'][la1,0,la3],'pa':ldic['extShear PA'][la1,0,la3]})
lenses = [lens1,shear]


colours = ['F606W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = xoffset
        dy = yoffset
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dx,yo+dy
    image,sigma,psf = imgs[i],sigs[i],PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(srcs)+1),imin.size))
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=21)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n] = np.ones(model[n].shape)
    n +=1
    mmodel = model.reshape((n,image.shape[0],image.shape[1]))
    mmmodel = np.empty(((len(srcs)+1),image[mask].size))
    for m in range(mmodel.shape[0]):
        mmmodel[m] = mmodel[m][mask]
    op = (mmmodel/sigma[mask]).T
    rhs = image[mask]/sigma[mask]
    print image[mask].size, image.size, image[~mask].size
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
    comps = False
    if comps == True:
        CotSomponents(components,colours[i])
    fits.append(fit)



pl.figure()
pl.plot(lp[:,0])
