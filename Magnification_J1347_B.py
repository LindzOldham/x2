import pylab as pl, numpy as np, pyfits
import lensModel2
from imageSim import SBModels,convolve
from pylens import *
import indexTricks as iT
import numpy
from scipy.interpolate import RectBivariateSpline, splrep, splev, splint
from scipy import optimize

result = np.load('/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_TWO')
lp= result[0]
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]
det = result[2]
dx,dy = det['xoffset'][a1,a2], det['yoffset'][a1,a2]

srcs,gals,lenses = [],[],[]
srcs.append(SBModels.Sersic('Source 1', {'x':det['Source 2 x'][a1,a2],'y':det['Source 2 y'][a1,a2],'q':det['Source 1 q'][a1,a2],'pa':det['Source 1 pa'][a1,a2],'re':det['Source 1 re'][a1,a2],'n':det['Source 1 n'][a1,a2]}))
srcs.append(SBModels.Sersic('Source 2', {'x':det['Source 2 x'][a1,a2],'y':det['Source 2 y'][a1,a2],'q':det['Source 2 q'][a1,a2],'pa':det['Source 2 pa'][a1,a2],'re':det['Source 2 re'][a1,a2],'n':det['Source 2 n'][a1,a2]}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':det['Galaxy 1 x'][a1,a2],'y':det['Galaxy 1 y'][a1,a2],'q':det['Galaxy 1 q'][a1,a2],'pa':det['Galaxy 1 pa'][a1,a2],'re':det['Galaxy 1 re'][a1,a2],'n':det['Galaxy 1 n'][a1,a2]}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':det['Lens 1 x'][a1,a2],'y':det['Lens 1 y'][a1,a2],'q':det['Lens 1 q'][a1,a2],'pa':det['Lens 1 pa'][a1,a2],'b':det['Lens 1 b'][a1,a2],'eta':det['Lens 1 eta'][a1,a2]}))
lenses.append(MassModels.ExtShear('shear',{'x':det['Lens 1 x'][a1,a2],'y':det['Lens 1 y'][a1,a2],'b':det['extShear'][a1,a2], 'pa':det['extShear PA'][a1,a2]}))

img1 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci_cutout.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_noisemap.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf.fits')[0].data.copy()
psf1 = psf1[15:-15,15:-15]
psf1 /= psf1.sum()

img2 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci_cutout.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_noisemap.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_psf_#2.fits')[0].data.copy()
psf2 = psf2[15:-15,15:-16]
psf2 /= psf2.sum()

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]


PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)

mask = np.zeros(img1.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(yc,xc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

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
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    comps.append(components)
    models.append(model)
    fits.append(fit)

mag_ims = []
mag_srcs = []
mus = []
ZPs = [26.493,25.95]
cols = ['F606W', 'F814W']
for i in range(len(imgs)):
    model = comps[i]
    galaxy = model[0] # this has ONE galaxy component
    source = model[1] + model[2]
    ZP = ZPs[i]
    mag_im = -2.5*np.log10(np.sum(source)) + ZP
    mag_ims.append(mag_im)
    img = imgs[i]
    galsub = img-galaxy
    #pyfits.writeto('/data/ljo31/Lens/J1323/galsub_'+cols[i]+'.fits',galsub,clobber=True)
    m1,m2 = srcs[0].getMag(fits[i][1], ZP), srcs[1].getMag(fits[i][2],ZP)
    print m1,m2
    F = 10**(0.4*(ZP-m1)) + 10**(0.4*(ZP-m2))
    mag_src = -2.5*np.log10(F) + ZP
    mag_srcs.append(mag_src)
    mu = 10**(0.4*(mag_src-mag_im))
    mus.append(mu)
    print 'mag_lensed =' , '%.2f'%mag_im, 'mag_intrinsic=', '%.2f'%mag_src, '%.2f'%mu
                                                           
### may as well also measure size of source here
## evaluate both sources over a grid and add them together. Then spline them along their radial coordinate to get the 1D SB and hence the effective radius.
Xgrid = np.logspace(-4,5,1501)
Ygrid = np.logspace(-4,5,1501)
for i in range(len(imgs)):
    source = fits[i][1]*srcs[0].eval(Xgrid) + fits[i][2]*srcs[1].eval(Xgrid)
    R = Xgrid.copy()
    #pl.figure()
    #pl.loglog(R,source)
    light = source*2.*np.pi*R
    mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
    intlight = np.zeros(len(R))
    for i in range(len(R)):
        intlight[i] = splint(0,R[i],mod)
    model = splrep(intlight[:-300],R[:-300])
    reff = splev(0.5*intlight[-1],model)
    #pl.figure()
    #pl.loglog(R,intlight)
    #pl.loglog(splev(intlight,model),intlight)
    print 'reff in kpc and arcsec', reff, reff*0.05*6.915
    # source redshift is 0.63 - scale is 6.915 kpc/"
    
