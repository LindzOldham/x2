import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl

# run number
X=9
print X
Y = 1
print Y

# plot things
def NotPlicely(image,im):
    ext = [0,image.shape[0],0,image.shape[1]]
    pl.figure()
    pl.subplot(131)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext)
    pl.colorbar()
    pl.title('data')
    pl.subplot(132)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext)
    pl.colorbar()
    pl.title('model')
    pl.subplot(133)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext)
    pl.colorbar()
    pl.title('data-model')
    pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')



pref = 'firstUpdate'

imgName = '/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_sci_cutout_2.fits'
sigName = '/data/ljo31/Lens/J1606/dump/SDSSJ1606+2235_F606W_noise_cutout_2.fits'
sigName = '/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_noisemap.fits'
psfName = '/data/ljo31/Lens/J1606/SDSSJ1606+2235_F606W_psf.fits'

img2Name = '/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_sci_cutout_2.fits'
sig2Name = '/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_noisemap.fits'
psf2Name = '/data/ljo31/Lens/J1606/SDSSJ1606+2235_F814W_psf.fits'

guiFile = '/data/ljo31/Lens/J1606/fits/ModelFit20_uniform'


img1 = pyfits.open(imgName)[0].data.copy()
sig1 = pyfits.open(sigName)[0].data.copy()
psf1 = pyfits.open(psfName)[0].data.copy()

img2 = pyfits.open(img2Name)[0].data.copy()
sig2 = pyfits.open(sig2Name)[0].data.copy()
psf2 = pyfits.open(psf2Name)[0].data.copy()

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
yc,xc = iT.overSample(img1.shape,1.)
for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)


OVRS = 1

G,L,S,offsets,_ = numpy.load(guiFile)

pars = []
cov = []
### first parameters need to be the offsets
xoffset = offsets[0][3]
yoffset = offsets[1][3]
pars.append(pymc.Uniform('xoffset',-5.,5.,value=offsets[0][3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=offsets[1][3]))
cov += [2,2]

srcs = []

for name in S.keys():
    s = S[name]
    p = {}
    for key in 'x','y','q','pa','re','n':
        if s[key]['type']=='constant':
            p[key] = s[key]['value']
        else:
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            #print name,'&', key,'&', '%.1f'%lo,'&', '%.1f'%val,'&', '%.1f'%hi,'\\'
            if key == 'pa' and name == 'Source 2':
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=100))
            #elif key == 'x' and name == 'Source 1':
            #    
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            #pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
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
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=150))
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
    lp = 0.
    for i in range(len(imgs)):
        if i == 0:
            x0,y0 = 0,0
        else:
            x0 = pars[0].value # xoffset
            y0 = pars[1].value # yoffset
            #print x0,y0
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        lp += lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,1,
                                verbose=False,psf=psf,csub=1)
    return lp
   

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
    S.sample(4*len(pars)**2)

    #S = AMAOpt(pars,[likelihood],[logP],cov=optCov/8.)
    #S.set_minprop(len(pars)*2)
    #S.sample(10*len(pars)**2)

    #S = AMAOpt(pars,[likelihood],[logP],cov=optCov/8.)
    #S.set_minprop(len(pars)*2)
    #S.sample(10*len(pars)**2)


logp,trace,det = S.result() # log likelihoods; chain (steps * params); det['extShear PA'] = chain in this variable

## plot some chains
import triangle

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

#for i in range(len(keylist)):
#    if key[0:-2] == 'Source 1':
#        print keylist[i], 'is', dkeylist[i]
#for i in range(len(keylist)):
#    if key[0:-2] == 'Source 2':
#        print keylist[i], 'is', dkeylist[i]



pl.figure()
pl.plot(logp)
pl.title('log P')
pl.savefig('/data/ljo31/Lens/TeXstuff/logPrun'+str(X)+'.png')

ims = []
models = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = det['xoffset'][-1], det['yoffset'][-1] # xoffset, yoffset #
        print x0,y0
    im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True) # return loglikelihood
    im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True) # return the model decomposed into the separate galaxy and source components
    ims.append(im)
    models.append(model)

colours = ['F606W', 'F814W']
for i in range(len(imgs)):
    image = imgs[i]
    im = ims[i]
    model = models[i]
    sigma = sigs[i]
    pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1606/components_uniform'+str(colours[i])+'.fits',clobber=True)
    pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1606/model_uniform'+str(colours[i])+'.fits',clobber=True)
    pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1606/resid_uniform'+str(colours[i])+'.fits',clobber=True)
    #f = open('%s_result.dat'%pref,'wb')
    #cPickle.dump(coeff,f,2)
    #f.close()
    NotPlicely(image,im)

# investigate pa
#for key in det.keys():
#    if key == 'Source 1 pa' or key == 'Source 2 pa':
#        pl.figure()
#        pl.plot(det[key])
#        pl.title(key)
#        print key, det[key][-1]
#        pl.savefig('/data/ljo31/Lens/TeXstuff/PArun'+str(X)+'source'+str(key[7])+'.png')


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
print det['xoffset'], det['yoffset']
#print xoffset, yoffset
