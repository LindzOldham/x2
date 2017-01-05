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


img1 = py.open('/data/ljo31/Lens/J0837/F606W_sci_cutout_huge.fits')[0].data.copy()#[30:-30,30:-30] 
sig1 = py.open('/data/ljo31/Lens/J0837/F606W_noisemap_huge.fits')[0].data.copy()#[30:-30,30:-30] 
psf1 = py.open('/data/ljo31/Lens/J0837/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)
img2 = py.open('/data/ljo31/Lens/J0837/F814W_sci_cutout_huge.fits')[0].data.copy()#[30:-30,30:-30]
sig2 = py.open('/data/ljo31/Lens/J0837/F814W_noisemap_huge.fits')[0].data.copy()#[30:-30,30:-30] 
psf2 = py.open('/data/ljo31/Lens/J0837/F814W_psf3.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)
Dx,Dy = -100,-100
OVRS=1
mask = py.open('/data/ljo31/Lens/J0837/mask_huge.fits')[0].data


result = np.load('/data/ljo31/Lens/LensModels/twoband/J0837_211')
lp,trace,dic,_= result
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,xo=xc+Dx,xo+Dx
yc,yo=yc+Dy,yo+Dy
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2.T
mask2 = mask2==0
mask = mask==0

gals = []
for name in ['Galaxy 1','Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = np.median(dic[name+' '+key])
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            p[key] = np.median(dic[name+' '+key])
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

lenses = []
p = {}
for key in 'x','y','q','pa','b','eta':
    p[key] = np.median(dic['Lens 1 '+key])
lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = np.median(dic['extShear'])
p['pa'] = np.median(dic['extShear PA'])
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in ['Source 2','Source 1']:
    p = {}
    if name == 'Source 2':
        for key in 'q','re','n','pa':
            p[key] = np.median(dic[name+' '+key])
        for key in 'x','y': 
            p[key] = np.median(dic[name+' '+key])# + lenses[0].pars[key]
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            p[key] = np.median(dic[name+' '+key])
        for key in 'x','y':
            p[key] = np.median(dic[name+' '+key]) + lenses[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))



galsubs = []
models,comps = [],[]
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = np.median(dic['xoffset'])
        dy = np.median(dic['yoffset'])
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=23) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=23)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        if src.name == 'Source 2':
            model[n] *= -1
            print src.name
        n +=1
    model[n]=np.ones(model[n].size)
    n+=1
    mmodel = model.reshape((n,image.shape[0],image.shape[1]))
    mmmodel = np.empty(((len(gals) + len(srcs)+1),image[mask].size))
    for m in range(mmodel.shape[0]):
        mmmodel[m] = mmodel[m][mask]
    op = (mmmodel/sigma[mask]).T
    rhs = image[mask]/sigma[mask]
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    n = 0
    galim = image.copy()
    for n in range(len(gals)):
        galim -=components[n]
    galsubs.append(galim)
    models.append(model)
    comps.append(components)
    pl.figure()
    pl.subplot(211)
    pl.imshow(galim,interpolation='nearest',origin='lower',cmap='jet',vmin=-0.5,vmax=0.5)
    pl.colorbar()
    pl.subplot(212)
    pl.imshow(np.sum(components,0),interpolation='nearest',origin='lower',cmap='jet')
    pl.colorbar()
    pl.show()

    py.writeto('/data/ljo31/Lens/J0837/galsub_new_meds_'+str(i)+'.fits',galim,clobber=True)

