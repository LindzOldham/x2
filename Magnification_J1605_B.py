import pylab as pl, numpy as np, pyfits
import lensModel2
from imageSim import SBModels,convolve
from pylens import *
import indexTricks as iT
import numpy
from scipy.interpolate import RectBivariateSpline, splrep, splev, splint
from scipy import optimize
import pyfits as py

### this currently isn't working because it's computing fits wrongly. Fix this at some point when it becomes important...


# this is from the optimiser, so need to account for its different shape
guiFile = '/data/ljo31/Lens/J1605/terminal_iterated_4'
trace = numpy.load('/data/ljo31/Lens/J1605/traceFINAL.npy')
lp = numpy.load('/data/ljo31/Lens/J1605/logPFINAL.npy')
det = numpy.load('/data/ljo31/Lens/J1605/detFINAL.npy')[()]
dx,dy = det['xoffset'][-1], det['yoffset'][-1]

srcs,gals,lenses = [],[],[]
srcs.append(SBModels.Sersic('Source 1', {'x':det['Source 1 x'][-1],'y':det['Source 1 y'][-1],'q':det['Source 1 q'][-1],'pa':det['Source 1 pa'][-1],'re':det['Source 1 re'][-1],'n':det['Source 1 n'][-1]}))
srcs.append(SBModels.Sersic('Source 2', {'x':det['Source 1 x'][-1],'y':det['Source 1 y'][-1],'q':det['Source 2 q'][-1],'pa':det['Source 2 pa'][-1],'re':det['Source 2 re'][-1],'n':det['Source 2 n'][-1]}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':det['Galaxy 1 x'][-1],'y':det['Galaxy 1 y'][-1],'q':det['Galaxy 1 q'][-1],'pa':det['Galaxy 1 pa'][-1],'re':det['Galaxy 1 re'][-1],'n':det['Galaxy 1 n'][-1]}))
gals.append(SBModels.Sersic('Galaxy 2', {'x':det['Galaxy 1 x'][-1],'y':det['Galaxy 1 y'][-1],'q':det['Galaxy 2 q'][-1],'pa':det['Galaxy 2 pa'][-1],'re':det['Galaxy 2 re'][-1],'n':det['Galaxy 2 n'][-1]}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':det['Lens 1 x'][-1],'y':det['Lens 1 y'][-1],'q':det['Lens 1 q'][-1],'pa':det['Lens 1 pa'][-1],'b':det['Lens 1 b'][-1],'eta':det['Lens 1 eta'][-1]}))
lenses.append(MassModels.ExtShear('shear',{'x':det['Lens 1 x'][-1],'y':det['Lens 1 y'][-1],'b':det['extShear'][-1], 'pa':det['extShear PA'][-1]}))

img1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2_masked.fits')[0].data.copy() # masking out the possiblyasecondsource regions!
psf1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf2= psf2[15:-16,14:-16]
psf2 /= psf2.sum()


imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,yc,xo,yo = xc-15,yc-15,xo-15,yo-15

mask = np.ones(img1.shape)
tck = RectBivariateSpline(xo[0],yo[:,0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2.T
mask2 = mask2==1
mask = mask==1

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

fits = []
comps = []
models = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    print psf.shape, sigma.shape,image.shape
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = dx,dy
    xp,yp = xc+x0,yc+y0
    #print xp,yp
    imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
    n = 0
    model = np.empty(((len(gals) + len(srcs)),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    print model.shape
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    print fit, chi
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    comps.append(components)
    models.append(model)
    fits.append(fit)

mag_ims = []
mag_srcs = []
mus = []
ZPs = [25.711,25.95]
cols = ['F555W', 'F814W']
for i in range(len(imgs)):
    model = comps[i]
    galaxy = model[0] + model[1]
    source = model[2] + model[3]
    ZP = ZPs[i]
    mag_im = -2.5*np.log10(np.sum(source)) + ZP
    mag_ims.append(mag_im)
    img = imgs[i]
    galsub = img-galaxy
    #pyfits.writeto('/data/ljo31/Lens/J1323/galsub_'+cols[i]+'.fits',galsub,clobber=True)
    m1,m2 = srcs[0].getMag(fits[i][2], ZP), srcs[1].getMag(fits[i][3],ZP)
    F = 10**(0.4*(ZP-m1)) + 10**(0.4*(ZP-m2))
    mag_src = -2.5*np.log10(F) + ZP
    mag_srcs.append(mag_src)
    mu = 10**(0.4*(mag_src-mag_im))
    mus.append(mu)
    print 'mag_lensed = ', mag_im, 'mag_intrinsic = ', mag_src, 'magnification = ', mu
                                                           
### may as well also measure size of source here
## evaluate both sources over a grid and add them together. Then spline them along their radial coordinate to get the 1D SB and hence the effective radius.
Xgrid = np.logspace(-4,5,1501)
Ygrid = np.logspace(-4,5,1501)
for i in range(len(imgs)):
    source = fits[i][2]*srcs[0].eval(Xgrid) + fits[i][3]*srcs[1].eval(Xgrid)
    R = Xgrid.copy()
    #pl.figure()
    #pl.loglog(R,source)
    light = source*2.*np.pi*R
    mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
    intlight = np.zeros(len(R))
    for i in range(len(R)):
        intlight[i] = splint(0,R[i],mod)
    model = splrep(intlight[:-200],R[:-200])
    reff = splev(0.5*intlight[-1],model)
    #pl.figure()
    #pl.loglog(R,intlight)
    #pl.loglog(splev(intlight,model),intlight)
    print 'reff in kpc and arcsec', reff, reff*0.05*6.435
    # source redshift is 0.689 0 scale is 7.187 kpc / '' and 0.05 arcsec/pixel 
    
