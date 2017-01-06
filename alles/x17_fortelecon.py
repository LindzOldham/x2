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
import SBBModels, SBBProfiles


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

def CotSomponents(components,col):
    pl.figure()
    pl.subplot(221)
    pl.imshow(components[0],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('galaxy 1 ')
    pl.subplot(222)
    pl.imshow(components[1],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('galaxy 2 ')
    pl.subplot(223)
    pl.imshow(components[2],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('source 1 ')
    pl.subplot(224)
    pl.imshow(components[3],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('source 2 ')
    pl.suptitle(col)

img1 = py.open('/data/ljo31/Lens/J1446/F606W_sci_cutout.fits')[0].data.copy()#[20:-20,:-5]
sig1 = py.open('/data/ljo31/Lens/J1446/F606W_noisemap.fits')[0].data.copy()#[20:-20,:-5]
psf1 = py.open('/data/ljo31/Lens/J1446/F606W_psf1.fits')[0].data.copy()
psf1 = psf1[5:-5,5:-5]
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1446/F814W_sci_cutout.fits')[0].data.copy()#[20:-20,:-5]
sig2 = py.open('/data/ljo31/Lens/J1446/F814W_noisemap.fits')[0].data.copy()#[20:-20,:-5]
psf2 = py.open('/data/ljo31/Lens/J1446/F814W_psf1.fits')[0].data.copy()
psf2 = psf2[6:-6,7:-7]
psf2 = psf2/np.sum(psf2)

result = np.load('/data/ljo31/Lens/J1446/emcee17')

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
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
yc,yo=yc-20.,yo-20.
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
        gals.append(SBBModels.Sersic(name,p))
    elif name == 'Galaxy 2':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
        #for key in 'x','y':
        #    p[key] = gals[0].pars[key]
        gals.append(SBBModels.Sersic(name,p))

print gals
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
for name in ['Source 2','Source 1']:
    p = {}
    print name
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            #p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
            p[key] = srcs[0].pars[key]
    srcs.append(SBBModels.Sersic(name,p))


colours = ['F606W', 'F814W']
models = []
fits = []
galsubs = []
maxes = [0.2,0.6]
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
    galaxy = np.empty((len(gals),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=23)
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        #pl.figure()
        #pl.imshow(tmp)
        #pl.colorbar()
        model[n] = tmp.ravel()
        galaxy[n] = tmp.ravel()
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
        n +=1
    model[n] = np.ones(model[n].shape)
    n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    #NotPlicely(image,model,sigma)
    #pl.suptitle(str(colours[i]))
    #pl.show()
    comps = False
    if comps == True:
        CotSomponents(components,colours[i])
    fits.append(fit)
    galsub = image - galaxy[0].reshape(image.shape)*fit[0] - galaxy[1].reshape(image.shape)*fit[1]
    pl.figure()
    pl.imshow(galsub, interpolation='nearest',origin='lower',vmin=0,vmax=maxes[i])
    pl.title('galaxy subtracted image, '+str(colours[i]))
    pl.colorbar()
    py.writeto('/data/ljo31/Lens/J1446/galsub_'+str(colours[i])+'.fits',galsub,clobber=True)
    pl.figure()
    pl.title('source plane: '+str(colours[i]))
    source = srcs[0].pixeval(xp,yp)*fit[2] + srcs[1].pixeval(xp,yp)*fit[3]
    pl.imshow(source, interpolation='nearest',origin='lower',vmin=0,vmax=maxes[i])
    pl.colorbar()
    pl.figure()
    pl.subplot(121)
    pl.imshow(srcs[0].pixeval(xp,yp)*fit[2], interpolation='nearest',origin='lower',vmin=0,vmax=maxes[i])
    pl.colorbar()
    pl.title('component 1')
    pl.subplot(122)
    pl.imshow(srcs[1].pixeval(xp,yp)*fit[3], interpolation='nearest',origin='lower',vmin=0,vmax=maxes[i])
    pl.colorbar()
    pl.title('component 2')
