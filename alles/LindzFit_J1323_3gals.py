import cPickle,numpy,pyfits
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

''' X = 22 -  long parallel tempered
X = 23 - long single tempered
X = 24 - like 23, only trying to start source 2 off with less ellipticity. q2 = 0.8
X = 25 - like 23, only roatitng galaxy 1 by 90 degrees to see if this helps with q = 1.
X = 26 - try increasing cov for q> Make source 2 slightly bigger and less elliptical by hand? Keep galaxy 1 rotated.
X = 27 - bm25_fancy. iterated 25, but then tried changing source 2 a bit! - didn't work out well.
X = 28 - bm25_iterated, with a mask over the probable dust lane. This is a bit risky as it's so close-in to the arcs, but we can have a look.
X = 29 - bm25_fancy2, using MASK3 - didn't work out well.
X = 30 - see readboth.py
X = 32 - bm_iterated for a long time
X = 33 - 26, but for longer
X = 34 - bm25_a
X = 35 - as 34, but with parallel tempering for ein bisschen.
X = 36 - bm25_c - rotate source to 130 degrees and make it less elliptical
X = 37 - bm25_e
X = 38 - bm25_f
X = 39 - bm25_g
X = 40 - bm25_h
X = 41 - bm25_i
X = 42 - 3gals2
X = 43 - 3gals4
X = 45 - 3gals2, only running for longer. Because X  = 42 was still increasing its logp.
'''

X = 45


print X

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
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
    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')

img1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci_cutout.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_noisemap.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf2.fits')[0].data.copy()
psf1 = psf1[10:-10,11:-10] # possibly this is too small? See how it goes
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci_cutout.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_noisemap.fits')[0].data.copy()
#psf2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf2.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf3.fits')[0].data.copy()
psf2 = psf2[8:-8,9:-8]
psf2 /= psf2.sum()


guiFile = '/data/ljo31/Lens/J1323/bm13_iterated'
#guiFile = '/data/ljo31/Lens/J1323/bm25_fancy2'
guiFile = '/data/ljo31/Lens/J1323/bm25_a'
guiFile = '/data/ljo31/Lens/J1323/bm25_c'
guiFile = '/data/ljo31/Lens/J1323/bm25_e'
guiFile = '/data/ljo31/Lens/J1323/bm25_f'
guiFile = '/data/ljo31/Lens/J1323/bm25_g'
guiFile = '/data/ljo31/Lens/J1323/bm25_h'
guiFile = '/data/ljo31/Lens/J1323/bm25_i'
guiFile = '/data/ljo31/Lens/J1323/3gals2'
#guiFile = '/data/ljo31/Lens/J1323/3gals4'



print guiFile

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 3 
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)

mask = pyfits.open('/data/ljo31/Lens/J1323/mask2_F814W.fits')[0].data.copy()
#mask = pyfits.open('/data/ljo31/Lens/J1323/mask3_F814W.fits')[0].data.copy()
#mask = mask==0
# for now, don't use a mask
#mask = np.zeros(img1.shape)
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

G,L,S,offsets,shear = numpy.load(guiFile)

pars = []
cov = []
### first parameters need to be the offsets
xoffset = offsets[0][3]
yoffset = offsets[1][3]
pars.append(pymc.Uniform('xoffset',-5.,5.,value=offsets[0][3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=offsets[1][3]))
cov += [0.4,0.4]


srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        for key in 'x','y','q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                if key == 're':
                    pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*1)
                elif key == 'q':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*1)
                elif key == 'n':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*1)
                elif key == 'pa': # to get no degeneracy
                    if val >0:
                        pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                    elif val<0:
                        pars.append(pymc.Uniform('%s %s'%(name,key),lo,0,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*10)
                else:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                if key == 're':
                    pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*1)
                elif key == 'q':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*1)
                elif key == 'n':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*1)
                elif key == 'pa':
                    if val >0:
                        pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                    elif val<0:
                        pars.append(pymc.Uniform('%s %s'%(name,key),lo,0,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev']*10)
                else:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                    p[key] = pars[-1]
                    cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))


gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                if key == 'pa':
                    if val >0:
                        pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                    elif val<0:
                        pars.append(pymc.Uniform('%s %s'%(name,key),lo,0,value=val))
                elif key == 'q':
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                else:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    elif name == 'Galaxy 3':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))


lenses = []
for name in L.keys():
    s = L[name]
    p = {}
    for key in 'x','y','q','pa','b','eta':
        if s[key]['type']=='constant':
            p[key] = s[key]['value']
        else:
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            if key == 'pa':
                if val >0:
                    pars.append(pymc.Uniform('%s %s'%(name,key),0,hi,value=val))
                elif val<0:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,0,value=val))
                cov.append(s[key]['sdev']*10)
            elif key == 'q':
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                cov.append(s[key]['sdev']*1)
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                cov.append(s[key]['sdev'])
            p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=shear[0]['b']['value']))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=shear[0]['pa']['value']))
cov.append(10.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

print len(pars), len(cov)
for p in pars:
    print p

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    models = []
    for i in range(len(imgs)):
        if i == 0:
            dx,dy = 0,0
        else:
            dx = pars[0].value 
            dy = pars[1].value 
        xp,yp = xc+dx,yc+dy
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
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
        model = (model.T*fit).sum(1)
        resid = (model-imin)/sigin
        lp += -0.5*(resid**2.).sum()
        models.append(model)
    return lp #,models

   
@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp #[0]

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)


#S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=6,nwalkers=60,ntemps=4)
#S.sample(1000)
S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=6,nwalkers=100)
S.sample(4500)

outFile = '/data/ljo31/Lens/J1323/emcee'+str(X)
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

result = S.result()
lp = result[0]



trace = numpy.array(result[1])
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)


## now we need to interpret these resultaeten
logp,coeffs,dic,vals = result
ii = np.where(logp==np.amax(logp))
coeff = coeffs[ii][0]

colours = ['F555W', 'F814W']
#mods = S.blobs
models = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
    if i == 0:
        dx,dy = 0,0
    else:
        dx = pars[0].value 
        dy = pars[1].value 
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp = src.pixeval(x0,y0,1./OVRS,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    #for j in range(len(components)):
    #    pl.figure()
    #    pl.imshow(components[j])
    #    pl.colorbar()
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(X)+str(colours[i]))
    pl.show()

