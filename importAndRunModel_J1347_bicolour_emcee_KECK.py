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
from imageSim import SBModels,convolve,SBObjects


def MakeCuts():
    image = pyfits.open('/data/mauger/EELs/SDSSJ1347-0101/J1347_med.fits')[0].data.copy()[905:1025,910:1030]
    header = pyfits.open('/data/mauger/EELs/SDSSJ1347-0101/J1347_med.fits')[0].header
    pyfits.writeto('/data/ljo31/Lens/J1347/J1347_med_cutout.fits',image,header,clobber=True)

def MakeMaps():
    image = pyfits.open('/data/mauger/EELs/SDSSJ1347-0101/J1347_med.fits')[0].data.copy()
    header = pyfits.open('/data/mauger/EELs/SDSSJ1347-0101/J1347_med.fits')[0].header
    cut1 = image[1035:1080,815:880]
    cut2 = image[970:1020,1065:1130]
    var1,var2 = np.var(cut1),np.var(cut2)
    poisson = np.mean((var1,var2))
    sigma = poisson**0.5
    im = pyfits.open('/data/ljo31/Lens/J1347/J1347_med_cutout.fits')[0].data.copy()
    smooth = ndimage.gaussian_filter(im,0.7)
    noisemap = np.where((smooth>0.7*sigma)&(im>0),im/120.+poisson, poisson)**0.5 # for now - nut find out the actual exposure time from Matt...
    pyfits.writeto('/data/ljo31/Lens/J1347/J137_med_sigma.fits',noisemap,header,clobber=True)
    pl.figure()
    pl.imshow(noisemap)
    pl.colorbar()



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
    pl.show()
    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')

image = pyfits.open('/data/ljo31/Lens/J1347/J1347_med_cutout.fits')[0].data.copy()
sigma = pyfits.open('/data/ljo31/Lens/J1347/J137_med_sigma.fits')[0].data.copy()
yc,xc = iT.coords(image.shape)*0.8 # wide camera: 0.04 arcsec/pixel

# Model the PSF as a Moffat
xp,yp = iT.coords((100,100))-50
OVRS = 1

# load up model
det = np.load('/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_TWO')[2]
lp = np.load('/data/ljo31/Lens/J1347/emcee_FINAL_uncertainties_TWO')[0]
ii = np.where(lp==np.amax(lp))

g1,l1,s1,s2,sh = {},{},{},{},{}

srcs = []
gals = []
lenses = []
coeff=[]
for name in det.keys():
    s = det[name]
    coeff.append(s[-1])
    if name[:8] == 'Source 1':
        s1[name[9:]] = s[ii][-1]
    elif name[:8] == 'Source 2':
        s2[name[9:]] = s[ii][-1]
    elif name[:6] == 'Lens 1':
        l1[name[7:]] = s[ii][-1]
    elif name[:8] == 'Galaxy 1':
        g1[name[9:]] = s[ii][-1]
    elif name[:8] == 'extShear':
        if len(name)<9:
            sh['b'] = s[ii][-1]
        elif name == 'extShear PA':
            sh['pa'] = s[ii][-1]
    
s1['x'] = s2['x'].copy()
s1['y'] = s2['y'].copy()
srcs.append(SBModels.Sersic('Source 1',s1))
srcs.append(SBModels.Sersic('Source 2',s2))
gals.append(SBModels.Sersic('Galaxy 1',g1))
lenses.append(MassModels.PowerLaw('Lens 1',l1))
sh['x'] = lenses[0].pars['x']
sh['y'] = lenses[0].pars['y']
lenses.append(MassModels.ExtShear('shear',sh))

pars = []
cov = []
## try some way of fitting!
#pars = [xoffset,yoffset,sig]
pars.append(pymc.Uniform('xoffset',9,13,value=10.5))
pars.append(pymc.Uniform('yoffset',10,14,value=12))
cov += [5,5] # think about this!
pars.append(pymc.Uniform('fwhm 1',0,8,value=4))
cov += [2]
pars.append(pymc.Uniform('q 1',0,1,value=0.7))
cov += [0.6]
pars.append(pymc.Uniform('pa 1',-180,180,value= 90 ))
cov += [50]
pars.append(pymc.Uniform('amp 1',0,1,value=0.7))
cov += [1]
pars.append(pymc.Uniform('fwhm 2',0,150,value= 15 )) 
cov += [10]
pars.append(pymc.Uniform('q 2',0,1,value=0.9))
cov += [0.6]
pars.append(pymc.Uniform('pa 2',-180,180,value= 90 )) 
cov += [50]
pars.append(pymc.Uniform('index 2',0,10,value= 4.5 )) 
cov += [2]
pars.append(pymc.Uniform('index 1',0,10,value= 4.5 ))
cov += [2]

@pymc.deterministic
def logP(value=0,p=pars):
    x0 = pars[0].value
    y0 = pars[1].value
    fwhm1 = pars[2].value.item()
    q1 = pars[3].value.item()
    pa1 = pars[4].value.item()
    amp1 = pars[5].value.item()
    fwhm2 = pars[6].value.item()
    q2 = pars[7].value.item()
    pa2 = pars[8].value.item()
    index2 = pars[9].value.item()
    index1 = pars[10].value.item()
    amp2 = 1.-amp1
    psfObj1 = SBObjects.Moffat('psf 1',{'x':0,'y':0,'fwhm':fwhm1,'q':q1,'pa':pa1,'amp':10,'index':index1})
    psfObj2 = SBObjects.Moffat('psf 2',{'x':0,'y':0,'fwhm':fwhm2,'q':q2,'pa':pa2,'amp':10,'index':index2})
    psf1 = psfObj1.pixeval(xp,yp) * amp1
    psf2 = psfObj2.pixeval(xp,yp) * amp2
    psf = psf1 + psf2
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    return lensModel.lensFit(None,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,1,
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

print 'about to sample'
S = myEmcee.Emcee(pars+[logP],cov=optCov,nthreads=2,nwalkers=60)
S.sample(1000)
print 'sampled'

outFile = '/data/ljo31/Lens/J1347/emcee_KECK'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

result = S.result()
lp = result[0]

trace = numpy.array(result[1])
a1,a2,a3 = numpy.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

## now we need to interpret these resultaeten
logp,coeffs,det,vals = result
ii = np.where(logp==np.amax(logp))
coeff = coeffs[ii][0]

keylist = []
dkeylist = []
chainlist = []
for key in det.keys():
    keylist.append(key)
    dkeylist.append(det[key][-1])
    chainlist.append(det[key])

sigma = pyfits.open('/data/ljo31/Lens/J1347/J1247_med_sigma.fits')[0].data.copy()
x0 = det['xoffset'][-1]
y0 = det['yoffset'][-1]
fwhm1 = det['fwhm 1'][-1]
q1 = det['q 1'][-1]
pa1 = det['pa 1'][-1]
amp1 = det['amp 1'][-1]
fwhm2 = det['fwhm 2'][-1]
q2 = det['q 2'][-1]
pa2 = det['pa 2'][-1]
index2 = det['index 2'][-1]
index1 = det['index 1'][-1]
amp2 = 1.-amp1
psfObj1 = SBObjects.Moffat('psf 1',{'x':0,'y':0,'fwhm':fwhm1,'q':q1,'pa':pa1,'amp':10,'index':index1})
psfObj2 = SBObjects.Moffat('psf 2',{'x':0,'y':0,'fwhm':fwhm2,'q':q2,'pa':pa2,'amp':10,'index':index2})
psf1 = psfObj1.pixeval(xp,yp) * amp1
psf2 = psfObj2.pixeval(xp,yp) * amp2
psf = psf1 + psf2
psf /= psf.sum()
psf = convolve.convolve(image,psf)[1]
im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
model = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True)

NotPlicely(image,im,sigma)
pl.savefig('/data/ljo31/Lens/J1347/Keckpsf.eps')

for key in det.keys():
    print key, det[key][-1]

print 'KECK - x & y & sigma 1 & sigma 2 & pa 1 & pa 2 & q 1 & q 2 & amp 1 & index 2 \\'
print '%.1f'%det['xoffset'][-1], '&', '%.1f'%det['yoffset'][-1], '&', '%.1f'%det['fwhm 1'][-1], '&', '%.1f'%det['fwhm 2'][-1], '&', '%.1f'%det['pa 1'][-1], '&', '%.1f'%det['pa 2'][-1], '&', '%.1f'%det['q 1'][-1], '&', '%.1f'%det['q 2'][-1], '&', '%.1f'%det['amp 1'][-1], '&', '%.1f'%det['index 2'][-1], '&', '%.1f'%det['index 1'][-1], '\\'
