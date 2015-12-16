import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt

pref = 'firstUpdate'

imgName = '/data/ljo31/Lens/SDSSJ1606+2235_F606W_sci_cutout.fits'
sigName = '/data/ljo31/Lens/SDSSJ1606+2235_F606W_noise3_cutout.fits'
psfName = '/data/ljo31/Lens/SDSSJ1606+2235_F606W_psf.fits'

img2Name = '/data/ljo31/Lens/SDSSJ1606+2235_F814W_sci_cutout.fits'
sig2Name = '/data/ljo31/Lens/SDSSJ1606+2235_F814W_noise3_cutout.fits'
psf2Name = '/data/ljo31/Lens/SDSSJ1606+2235_F814W_psf.fits'

guiFile = '/data/ljo31/Lens/ModelFit12'


image = pyfits.open(imgName)[0].data.copy()
sigma = pyfits.open(sigName)[0].data.copy()
psf = pyfits.open(psfName)[0].data.copy()

yc,xc = iT.overSample(image.shape,1.)
psf /= psf.sum()
psf = convolve.convolve(image,psf)[1]

OVRS = 1

G,L,S,_,_ = numpy.load(guiFile)

pars = []
srcs = []
cov = []
for name in S.keys():
    s = S[name]
    p = {}
    for key in 'x','y','q','pa','re','n':
        if s[key]['type']=='constant':
            p[key] = s[key]['value']
        else:
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    srcs.append(SBModels.Sersic(name,p))

gals = []
for name in G.keys():
    s = G[name]
    p = {}
    for key in 'x','y','q','pa','re','n':
        if s[key]['type']=='constant':
            p[key] = s[key]['value']
        else:
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
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
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=0))
cov.append(0.01)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=0.))
cov.append(5.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))


npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    return lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc,yc,1,
                                verbose=False,psf=psf,csub=1)

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)

#S = levMar(pars,resid)
#self.outPars = pars
#return
for i in range(1):
    S = AMAOpt(pars,[likelihood],[logP],cov=optCov/4.)
    S.set_minprop(len(pars)*2)
    S.sample(80*len(pars)**2)

    S = AMAOpt(pars,[likelihood],[logP],cov=optCov/8.)
    S.set_minprop(len(pars)*2)
    S.sample(100*len(pars)**2)

    S = AMAOpt(pars,[likelihood],[logP],cov=optCov/8.)
    S.set_minprop(len(pars)*2)
    S.sample(100*len(pars)**2)


logp,trace,det = S.result()
coeff = []
for i in range(len(pars)):
    coeff.append(trace[-1,i])

coeff = numpy.asarray(coeff)

pars = coeff
o = 'npars = ['
for i in range(pars.size):
    o += '%f,'%(pars)[i]
o = o[:-1]+"]"
print o
im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,psf=psf,verbose=True)
im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True)
model = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True,getModel=True)


pyfits.PrimaryHDU(model).writeto('%s_components.fits'%pref,clobber=True)
pyfits.PrimaryHDU(im).writeto('%s_model.fits'%pref,clobber=True)
pyfits.PrimaryHDU(image-im).writeto('%s_resid.fits'%pref,clobber=True)
f = open('%s_result.dat'%pref,'wb')
cPickle.dump(coeff,f,2)
f.close()
