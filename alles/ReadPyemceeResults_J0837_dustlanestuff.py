import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
#import myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline



# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0)#,vmax=1.35) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0)#,vmax=1.35) #,vmin=vmin,vmax=vmax)
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

    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')

def SotPleparately(image,im,sigma,col):
    ext = [0,image.shape[0],0,image.shape[1]]
    pl.figure()
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data - '+str(col))
    pl.figure()
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model - '+str(col))
    pl.figure()
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot')
    pl.title('signal-to-noise residuals - '+str(col))
    pl.colorbar()


img1 = py.open('/data/ljo31/Lens/J0837/F606W_sci_cutout.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]
sig1 = py.open('/data/ljo31/Lens/J0837/F606W_noisemap.fits')[0].data.copy()[30:-30,30:-30] #[[15:-15,15:-15]
psf1 = py.open('/data/ljo31/Lens/J0837/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J0837/F814W_sci_cutout.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]
sig2 = py.open('/data/ljo31/Lens/J0837/F814W_noisemap.fits')[0].data.copy()[30:-30,30:-30] #[[15:-15,15:-15]
psf2 = py.open('/data/ljo31/Lens/J0837/F814W_psf3.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)



result = np.load('/data/ljo31/Lens/J0837/emcee8')

lp= result[0]
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]
print lp.shape, trace.shape

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
#xc,xo=xc-60.,xo-60.
#yc,yo=yc-50.,yo-50.
#xc,xo=xc-45.,xo-45.
#yc,yo=yc-35.,yo-35.
xc,xo=xc-30.,xo-30.
yc,yo=yc-20.,yo-20.
mask = py.open('/data/ljo31/Lens/J0837/mask.fits')[0].data.copy()[30:-30,30:-30] #[15:-15,15:-15]# + py.open('/data/ljo31/Lens/J0837/mask7.fits')[0].data.copy()
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

### first parameters need to be the offsets
xoffset =  dic['xoffset'][a1,a2,a3]
yoffset = dic['yoffset'][a1,a2,a3]
#xoffset2, yoffset2 = dic['Kp-V xoffset'][a1,a2,a3], dic['Kp-V yoffset'][a1,a2,a3]

gals = []
for name in ['Galaxy 1','Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key]=gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

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

srcs = []
for name in ['Source 1','Source 2']:
    p = {}
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = dic[name+' '+key][a1,a2,a3]#+lenses[0].pars[key]
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
            #p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))



colours = ['F606W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
    if i == 0:
        dx,dy = 0,0
    else:
        dx = xoffset
        dy = yoffset
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
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    kk = 0
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=11)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        if kk == 1:
            model[n] *=-1
        kk +=1
        n +=1
    model[n] = np.ones(model[n].shape)
    n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    fits.append(fit)

print fits

s1 = srcs[0].pixeval(xp,yp)*fits[0][2]
s2 = srcs[1].pixeval(xp,yp)*fits[0][3]
pl.figure()
pl.imshow(s1,interpolation='nearest',origin='lower')
pl.colorbar()
pl.figure()
pl.imshow(s2,interpolation='nearest',origin='lower')
pl.colorbar()

# calculate extinction
fobs = s1-s2
fmod = s1.copy()

mask=np.where(s2>0.002,0,1)
mask=mask==0
#pl.figure()
#pl.imshow(mask)

tau = -np.log(fobs[mask]/fmod[mask])
Av = 1.09* tau
ext = np.sum(Av)/len(s2[mask])
print ext
area = len(s2)*0.05*6.349 # 6.369 kpc/arcsec, 0.05 arcsec/pixel
Lambda = 6.*1e-6
dustmass = area * ext / Lambda
print dustmass, np.log10(dustmass)

magnitude = srcs[0].getMag(fits[0][2], 26.493) # F606W
# find rest mag = 19.51, source mag = 20.75, log10(L_v) = 11.05
