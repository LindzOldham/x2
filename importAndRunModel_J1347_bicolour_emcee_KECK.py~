import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
import lensModel2

''' herer we are just sampling to get uncertainties '''


#this is with the bigger images. Have to be careful about adding things to the coordinates properly.
X = 'FINAL'

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

guiFile = '/data/ljo31/Lens/J1347/emcee_2'

print guiFile

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 4
yc,xc = iT.overSample(img1.shape,OVRS)
yc,xc = yc,xc
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
        for key in 'x','y','q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
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
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=shear[0]['b']['value']))
cov.append(0.01)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=shear[0]['pa']['value']))
cov.append(5.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))
print shear[0]['b']['value']
print shear[0]['pa']['value']
print gals, lenses, srcs

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    fits = []
    for i in range(len(imgs)):
        if i == 0:
            x0,y0 = 0,0
        else:
            x0 = pars[0].value 
            y0 = pars[1].value 
            #print x0,y0
        image = imgs[i]
        sigma = sigs[i]
        psf = PSFs[i]
        lp += lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,
                                verbose=False,psf=psf,csub=1)
        fits.append(lensModel2.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,verbose=False,psf=psf,csub=1,showAmps=True))
    return lp,fits
   

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp[0]

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)*1.
print len(pars)
S = myEmcee.Emcee(pars+[logP],cov=optCov,nthreads=1,nwalkers=60)
S.sample(5000)

outFile = '/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_THREE'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

outFile2 = '/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_THREE_BLOBS'
f = open(outFile2,'wb')
cPickle.dump(S.blobs,f,2)
f.close()
# fits[a][b][0][c]
# a = sample number (so up to 3000)
# b = walker number (so up to 60)
# c: 0 = logP, 1 = fits !

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


ims = []
models = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    print psf.shape, sigma.shape,image.shape
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = dic['xoffset'][ii][0], dic['yoffset'][ii][0]
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,psf=psf,verbose=True)
    im = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    ims.append(im)
    models.append(model)

colours = ['F606W', 'F814W']
for i in range(len(imgs)):
    image = imgs[i]
    im = ims[i]
    model = models[i]
    sigma = sigs[i]
    #pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1347/components_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1347/model_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1347/resid_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #f = open('/data/ljo31/Lens/J1347/coeff'+str(X),'wb')
    #cPickle.dump(coeff,f,2)
    #f.close()
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))


tracer = np.zeros((60*4500,26))
for i in range(26):
    arr = trace[500:,:,i]
    arr = np.ravel(arr)
    tracer[:,i] = arr

upperlower = map(lambda v: (v[1]-v[0],v[1],v[2]-v[1]),zip(*np.percentile(tracer,[16,50,84],axis=0)))
upperlower = np.array(upperlower)


dx,dy,x1,y1,q1,pa1,re1,n1,q2,pa2,re2,n2,x3,y3,q3,pa3,re3,n3,x4,y4,q4,pa4,b,eta,shear,shearpa = trace[a1,a2,:]
dxa,dya,x1a,y1a,q1a,pa1a,re1a,n1a,q2a,pa2a,re2a,n2a,x3a,y3a,q3a,pa3a,re3a,n3a,x4a,y4a,q4a,pa4a,ba,etaa,sheara,shearpaa = upperlower[:,0]
dxb,dyb,x1b,y1b,q1b,pa1b,re1b,n1b,q2b,pa2b,re2b,n2b,x3b,y3b,q3b,pa3b,re3b,n3b,x4b,y4b,q4b,pa4b,bb,etab,shearb,shearpab = upperlower[:,2]

print 'source 1 ', '& $', '%.2f'%x1, '_{-', '%.2f'%x1a, '}^{+', '%.2f'%x1b, '}$ & $', '%.2f'%y1,'_{-', '%.2f'%y1a, '}^{+', '%.2f'%y1b, '}$ & $','%.2f'%n1, '_{-', '%.2f'%n1a, '}^{+','%.2f'%n1b, '}$ & $','%.2f'%re1,'_{-', '%.2f'%re1a, '}^{+', '%.2f'%re1b, '}$ & $','%.2f'%q1,'_{-', '%.2f'%q1a, '}^{+', '%.2f'%q1b, '}$  & $','%.2f'%pa1, '_{-', '%.2f'%pa1a, '}^{+', '%.2f'%pa1b, '}$', r'\\'

print 'source 2 ', '& $', '%.2f'%x1, '_{-', '%.2f'%x1a, '}^{+', '%.2f'%x1b, '}$ & $', '%.2f'%y1,'_{-', '%.2f'%y1a, '}^{+', '%.2f'%y1b, '}$ & $','%.2f'%n2, '_{-', '%.2f'%n2a, '}^{+','%.2f'%n2b, '}$ & $','%.2f'%re2,'_{-', '%.2f'%re2a, '}^{+', '%.2f'%re2b, '}$ & $','%.2f'%q2,'_{-', '%.2f'%q2a, '}^{+', '%.2f'%q2b, '}$ & $','%.2f'%pa2, '_{-', '%.2f'%pa2a, '}^{+', '%.2f'%pa2b, '}$', r'\\\hline'

print 'galaxy 1 ', '& $', '%.2f'%x3, '_{-', '%.2f'%x3a, '}^{+', '%.2f'%x3b, '}$ & $', '%.2f'%y3,'_{-', '%.2f'%y3a, '}^{+', '%.2f'%y3b, '}$ & $','%.2f'%n3, '_{-', '%.2f'%n3a, '}^{+','%.2f'%n3b, '}$ & $','%.2f'%re3,'_{-', '%.2f'%re3a, '}^{+', '%.2f'%re3b, '}$ & $','%.2f'%q3,'_{-', '%.2f'%q3a, '}^{+', '%.2f'%q3b, '}$ & $','%.2f'%pa3, '_{-', '%.2f'%pa3a, '}^{+', '%.2f'%pa3b, '}$', r'\\\hline'

print 'lens 1 ', '& $', '%.2f'%x4, '_{-', '%.2f'%x4a, '}^{+', '%.2f'%x4b, '}$ & $', '%.2f'%y4,'_{-', '%.2f'%y4a, '}^{+', '%.2f'%y4b, '}$ & $','%.2f'%eta, '_{-', '%.2f'%etaa, '}^{+','%.2f'%etab, '}$ & $','%.3f'%b,'_{-', '%.3f'%ba, '}^{+', '%.3f'%bb, '}$ & $','%.2f'%q4,'_{-', '%.2f'%q4a, '}^{+', '%.2f'%q4b, '}$ & $','%.2f'%pa4, '_{-', '%.2f'%pa4a, '}^{+', '%.2f'%pa4b, '}$' , r'\\\hline'

# also need to get uncertainties on the magnitudes now that we have the blobs...

blobs = S.blobs

fits = []
for i in range(60):
    for j in range(5000):
        fits.append(blobs[j][i][0][1])

fits = np.array(fits)

# distributions of magnitudes
mags = []
mags814 = []
from itertools import product
for i in range(fits.shape[0]):
    mags.append([[gals[0].getMag(fits[i,0,0],26.5), srcs[0].getMag(fits[i,0,1],26.5), srcs[1].getMag(fits[i,0,2],26.5)], [gals[0].getMag(fits[i,1,0],25.95), srcs[0].getMag(fits[i,1,1],25.95), srcs[1].getMag(fits[i,1,2],25.95)]])

mags = np.array(mags)

#mags606, mags814 = np.array(mags606), np.array(mags814)

outFile3 = '/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_THREE_MAGS'
f = open(outFile3,'wb')
cPickle.dump(mags,f,2)
f.close()
