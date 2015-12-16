import cPickle,numpy,pyfits
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
import lensModel2

img1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci_cutout.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_noisemap.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf2.fits')[0].data.copy()
psf1 = psf1[10:-10,11:-10] # possibly this is too small? See how it goes
psf1 = psf1/np.sum(psf1)
img2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci_cutout.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_noisemap.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf3.fits')[0].data.copy()
psf2 = psf2[8:-8,9:-8]
psf2 /= psf2.sum()
guiFile = '/data/ljo31/Lens/J1323/bm13_iterated'
imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]
PSFs = []
OVRS=1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)

from scipy.interpolate import RectBivariateSpline
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

G,L,S,offsets,_ = numpy.load(guiFile)

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
                else:
                    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                   cov.append(s[key]['sdev']*100) 
                else:
                    cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                   cov.append(s[key]['sdev']*100) 
                else:
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
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                cov.append(s[key]['sdev']*100)
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                cov.append(s[key]['sdev'])
            p[key] = pars[-1]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=0.01))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=12))
cov.append(100.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

#print len(pars), len(cov)
#for p in pars:
#    print p

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]



def logP(p,value=0.):
    lp = 0.
    models = []
    for i in range(len(imgs)):
        if i == 0:
            dx,dy = 0,0
        else:
            dx = pars[0].value 
            dy = pars[1].value 
        print 'dx,dy = ', dx, dy
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
        print -0.5*(resid**2.).sum()
        lp += -0.5*(resid**2.).sum()
        models.append(model)
        #print lp
    return lp #,models

def logP2(p,value=0.):
    lp = 0.
    for i in range(len(imgs)):
        if i == 0:
            x0,y0 = 0,0
        else:
            x0 = pars[0].value 
            y0 = pars[1].value
        print 'dx,dy = ', x0, y0
            #print x0,y0
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        lp += lensModel2.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,verbose=False,psf=psf,mask=mask,csub=1,showAmps=False)
        #print lp
    return lp

lp1 = logP(p=pars)
lp2 = logP2(p=pars)

print lp1, lp2
