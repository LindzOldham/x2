import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl

# run number
X=9
#print X
Y = 1
#print Y

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = amin(image), amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext) #,vmin=-0.25,vmax=0.25)
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext)#,vmin=-5,vmax=5)
    pl.title('signal-to-noise residuals')
    pl.colorbar()
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')



img = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#7.fits')[0].data.copy()         
image = img.copy()
sigma = sig.copy()
guiFile = '/data/ljo31/Lens/J1605/bm814_b'

yc,xc = iT.overSample(img.shape,1.)
yc,xc = yc-15.,xc-15.


psf /= psf.sum()
psf = convolve.convolve(image,psf)[1]

OVRS = 1

G,L,S,_,_ = numpy.load(guiFile)

pars = []
cov = []
srcs = []

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
            if key == 'pa':
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            else:
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
# use lensFit to calculate the likelihood at each point in the chain
for i in range(1):
    S = AMAOpt(pars,[likelihood],[logP],cov=optCov/4.)
    S.set_minprop(len(pars)*2)
    S.sample(100*len(pars)**2)

    #S = AMAOpt(pars,[likelihood],[logP],cov=optCov/8.)
    #S.set_minprop(len(pars)*2)
    #S.sample(10*len(pars)**2)

    #S = AMAOpt(pars,[likelihood],[logP],cov=optCov/8.)
    #S.set_minprop(len(pars)*2)
    #S.sample(10*len(pars)**2)


logp,trace,det = S.result() # log likelihoods; chain (steps * params); det['extShear PA'] = chain in this variable

coeff = []
for i in range(len(pars)):
    coeff.append(trace[-1,i])

coeff = numpy.asarray(coeff)

pars = coeff
o = 'npars = ['
for i in range(pars.size):
    o += '%f,'%(pars)[i]
o = o[:-1]+"]"

keylist = []
dkeylist = []
chainlist = []
for key in det.keys():
    keylist.append(key)
    dkeylist.append(det[key][-1])
    chainlist.append(det[key])

plot = False
if plot:
    for i in range(len(keylist)):
        pl.figure()
        pl.plot(chainlist[i])
        pl.title(str(keylist[i]))


pl.figure()
pl.plot(logp)
pl.title('log P')
#pl.savefig('/data/ljo31/Lens/TeXstuff/logPrun'+str(X)+'.png')

im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,psf=psf,verbose=True) # return loglikelihood
im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True) # return model
model = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True,getModel=True) #


#pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1605/components_uniform'+str(colours[i])+'.fits',clobber=True)
#pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1605/model_uniform'+str(colours[i])+'.fits',clobber=True)
#pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1605/resid_uniform'+str(colours[i])+'.fits',clobber=True)
#f = open('%s_result.dat'%pref,'wb')
#cPickle.dump(coeff,f,2)
#f.close()
NotPlicely(image,im,sigma)



### OUTPUT THE THINGS IN LATEX-FRIENDLY FORM!
#print '%.1f'%det['Source 1 x'][-1], '&', '%.1f'%det['Source 1 y'][-1], '&', '%.1f'%det['Source 1 n'][-1], '&', '%.1f'%det['Source 1 re'][-1], '&', '%.1f'%det['Source 1 q'][-1], '&', '%.1f'%det['Source 1 pa'][-1], '\\'
#print '%.1f'%det['Source 2 n'][-1], '&', '%.1f'%det['Source 2 re'][-1], '&', '%.1f'%det['Source 2 q'][-1], '&', '%.1f'%det['Source 2 pa'][-1], '\\'

#numpy.save('/data/ljo31/Lens/J1606/trace'+str(Y), trace)
#numpy.save('/data/ljo31/Lens/J1606/logP'+str(Y), logp)

for key in det.keys():
    print key, '%.1f'%det[key][-1]
print 'max lnL is ', max(logp)

#pl.figure()
#pl.imshow((im-image)/sigma)
#pl.colorbar()
#print xoffset, yoffset
