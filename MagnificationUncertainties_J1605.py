import pylab as pl, numpy as np, pyfits
import lensModel2
from imageSim import SBModels,convolve
from pylens import *
import indexTricks as iT
import numpy
from scipy.interpolate import RectBivariateSpline, splrep, splev, splint
from scipy import optimize
import itertools

### data starts
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

mask = np.zeros(img1.shape)
tck = RectBivariateSpline(xo[0],yo[:,0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2.T
mask2 = mask2==0
mask = mask==0

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)
### data ends
print 'data'

# this is from the optimiser, so need to account for its different shape
guiFile = '/data/ljo31/Lens/J1605/terminal_iterated_4'
trace = numpy.load('/data/ljo31/Lens/J1605/traceFINAL.npy')
lp = numpy.load('/data/ljo31/Lens/J1605/logPFINAL.npy')
det = numpy.load('/data/ljo31/Lens/J1605/detFINAL.npy')[()]
print 'here'
thin = lp [2000::10]
for key in det.keys():
    det[key] = det[key][2000::10]

MAG_IMS, MAG_SRCS, MU, REFFS, lps = [], [], [], [], []
for j in range(thin.shape[0]):
    lps.append(thin[i])
    dx,dy = det['xoffset'][j], det['yoffset'][j]
    srcs,gals,lenses = [],[],[]
    srcs.append(SBModels.Sersic('Source 1', {'x':det['Source 1 x'][j],'y':det['Source 1 y'][j],'q':det['Source 1 q'][j],'pa':det['Source 1 pa'][j],'re':det['Source 1 re'][j],'n':det['Source 1 n'][j]}))
    srcs.append(SBModels.Sersic('Source 2', {'x':det['Source 1 x'][j],'y':det['Source 1 y'][j],'q':det['Source 2 q'][j],'pa':det['Source 2 pa'][j],'re':det['Source 2 re'][j],'n':det['Source 2 n'][j]}))
    gals.append(SBModels.Sersic('Galaxy 1', {'x':det['Galaxy 1 x'][j],'y':det['Galaxy 1 y'][j],'q':det['Galaxy 1 q'][j],'pa':det['Galaxy 1 pa'][j],'re':det['Galaxy 1 re'][j],'n':det['Galaxy 1 n'][j]}))
    gals.append(SBModels.Sersic('Galaxy 2', {'x':det['Galaxy 1 x'][j],'y':det['Galaxy 1 y'][j],'q':det['Galaxy 2 q'][j],'pa':det['Galaxy 2 pa'][j],'re':det['Galaxy 2 re'][j],'n':det['Galaxy 2 n'][j]}))
    if trace.shape[-1] == 34:
        gals.append(SBModels.Sersic('Galaxy 3', {'x':det['Galaxy 1 x'][j],'y':det['Galaxy 1 y'][j],'q':det['Galaxy 3 q'][j],'pa':det['Galaxy 3 pa'][j],'re':det['Galaxy 3 re'][j],'n':det['Galaxy 3 n'][j]}))
    lenses.append(MassModels.PowerLaw('Lens 1', {'x':det['Lens 1 x'][j],'y':det['Lens 1 y'][j],'q':det['Lens 1 q'][j],'pa':det['Lens 1 pa'][j],'b':det['Lens 1 b'][j],'eta':det['Lens 1 eta'][j]}))
    lenses.append(MassModels.ExtShear('shear',{'x':det['Lens 1 x'][j],'y':det['Lens 1 y'][j],'b':det['extShear'][j], 'pa':det['extShear PA'][j]}))

    fits = []
    comps = []
    models = []
    for i in range(len(imgs)):
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
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
    #print fits
    #for i in range(np.ravel(fits).size):
    #    print i
    #    if fits[i] ==0.0:
    #        fits[i] = 1e-4
    #print fits
    mag_ims = []
    mag_srcs = []
    mus = []
    ZPs = [25.711,25.95]
    cols = ['F555W', 'F814W']
    for i in range(len(imgs)):
        model = comps[i]
        if trace.shape[-1] == 34:
            galaxy = model[0] + model[1] + model[2]
            source = model[3] + model[4]
        else:
            galaxy = model[0] + model[1]
            source = model[2] + model[3]
        ZP = ZPs[i]
        mag_im = -2.5*np.log10(np.sum(source)) + ZP
        mag_ims.append(mag_im)
        img = imgs[i]
        galsub = img-galaxy
        if trace.shape[-1] == 34:
            m1,m2 = srcs[0].getMag(fits[i][3], ZP), srcs[1].getMag(fits[i][4],ZP)
        else:
            m1,m2 = srcs[0].getMag(fits[i][2], ZP), srcs[1].getMag(fits[i][3],ZP)
        F = 10**(0.4*(ZP-m1)) + 10**(0.4*(ZP-m2))
        mag_src = -2.5*np.log10(F) + ZP
        mag_srcs.append(mag_src)
        mu = 10**(0.4*(mag_src-mag_im))
        mus.append(mu)
        #print mag_im, mag_src, mu
    MAG_SRCS.append(mag_srcs)
    MAG_IMS.append(mag_ims)
    MU.append(mus)
     ## evaluate both sources over a grid and add them together. Then spline them along their radial coordinate to get the 1D SB and hence the effective radius.
    Xgrid = np.logspace(-4,5,1501)
    Ygrid = np.logspace(-4,5,1501)
    Reff = []
    for i in range(len(imgs)):
        if trace.shape[-1] == 34:
            source = fits[i][3]*srcs[0].eval(Xgrid) + fits[i][4]*srcs[1].eval(Xgrid)
        else:
            source = fits[i][2]*srcs[0].eval(Xgrid) + fits[i][3]*srcs[1].eval(Xgrid)
        R = Xgrid.copy()
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
        #print reff, reff*0.05*6.435
        Reff.append(reff*0.05*6.435)
        # source redshift is 0.542 and 6.435 kpc/arcsec and 0.05 arcsec/pixel
        ## (lens redshift is 0.306)
    REFFS.append(Reff)

## then get uncertainties!
## need to separate the two bands
mi,ms,mu,reff = np.array(MAG_IMS), np.array(MAG_SRCS), np.array(MU), np.array(REFFS)
lps = np.array(lps)
ii = np.argmax(lps)
print 'these should be the maximum likelihoods...',  mi[ii],ms[ii],mu[ii],reff[ii]

grandarray_555 = np.column_stack((ms[:,0], mi[:,0], mu[:,0], reff[:,0]))
grandarray_814 = np.column_stack((ms[:,1], mi[:,1], mu[:,1], reff[:,1]))
lower5,mid5,upper5 = np.array(map(lambda v: (v[1]-v[0],v[1],v[2]-v[1]),zip(*np.percentile(grandarray_555,[16,50,84],axis=0)))).T
lower8,mid8,upper8 = np.array(map(lambda v: (v[1]-v[0],v[1],v[2]-v[1]),zip(*np.percentile(grandarray_814,[16,50,84],axis=0)))).T

#### print in a latex-readable form...
print r'band & intrinsic magnitude & lensed magnitude & magnification & $R_e$ (kpc) \\\hline'
print 'F555W & $', '%.2f'%ms[ii][0], '_{-', '%.2f'%lower5[0], '}^{+', '%.2f'%upper5[0], '}$ & $', '%.2f'%mi[ii][0], '_{-', '%.2f'%lower5[1], '}^{+', '%.2f'%upper5[1], '}$ & $', '%.2f'%mu[ii][0], '_{-', '%.2f'%lower5[2], '}^{+', '%.2f'%upper5[2], '}$ & $', '%.2f'%reff[ii][0], '_{-', '%.2f'%lower5[3], '}^{+', '%.2f'%upper5[3], r'}$ \\' 
print 'F814W & $', '%.2f'%ms[ii][1], '_{-', '%.2f'%lower8[0], '}^{+', '%.2f'%upper8[0], '}$ & $', '%.2f'%mi[ii][1], '_{-', '%.2f'%lower8[1], '}^{+', '%.2f'%upper8[1], '}$ & $', '%.2f'%mu[ii][1], '_{-', '%.2f'%lower8[2], '}^{+', '%.2f'%upper8[2], '}$ & $', '%.2f'%reff[ii][1], '_{-', '%.2f'%lower8[3], '}^{+', '%.2f'%upper8[3], r'}$ \\'

np.save('/data/ljo31/Lens/J1605/photFINAL_555', grandarray_555)
np.save('/data/ljo31/Lens/J1605/photFINAL_814', grandarray_814)
