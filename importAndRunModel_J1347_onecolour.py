import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import lensModel2

# try basicmodel4 with a load of different position angles for source 1
'''
X = 0 - one source component, one galaxy componnet. basicmodel2a
X = 1 - basicmodel2c. With two source components
X = 2 - basicmodel2e. With source 2 starting in a better position
X = 3 - basicmodel2f, with a pa = 80, q = 0.65 for source 2
X = 4 - basicmodel2g, with pa = 80, q = 0.75 for source 1
X = 5 - basicmodel2h, with source positions fixed. Based on basicmodelc.
X = 6 - basicmodel2h2 - rotated source 1
X = 7 - basicmodel2h3 - roated source 2 as well
X = 8 - basicmodel2h4 - focussing on re and n BEST SO FAR!!!
X = 10 - basicmodel2h4_alluniform
X = 11 - basicmodel2i
X = 12 - basicmodel2h4_alluniform_srcpas -  with pas of sources rotated by 90 degrees in case they want to be elliptical!
X = 13 trying to reprodice X = 8...
X=14
X = 15 - with X=8, pa = 140
X = 16 - putting a load of parameters in by hand
X = 17 - same as X=4 because this had the best starting logp, but this time adding in shear
X = 18 - as 17, but putting in re = 0.6 for source 1
X = 19- det8 (the guiFile version)!
X= 20 - as 19, but oversampling with OVRS = 8.
X = 21 - det8a. re=1, focus on shear
X = 22 - det8_alluniform - same as det8, only with positions also left free to varyscarymarycontrary. And back to OVRS = 4.
up next: as in 19 (so with OVRS = 4), but trying out different shears...
X = 23 - det8_alluniform. OVRS = 4. pa = 130, shear = 0.01 - ends up with shear = 0.0037, pa = 136, lp = -8956. Not good
X = 24 - det8_uniform with pa = 130, shear = -0.01 - ends up with shear = -0.003, pa = 127, lp = -8900. Also not good
X = 25 - det8_uniform rotating the sources by 90 degrees as seeing if they want to be elliptical. - shear  = 0.002, pa = 135, lp= -8862. Source ellipticities remain at 1 and 0.9.
X = 26 - with huge covariances on things - generally multiplied by 10, except for the PAs of sources and shear, which are multiplied by 100. I think this is necessary if we want to really explore the space!!!! REALLY BAD! Hardly stepped at all!
X = 27 - just increasing the covariances on the source and shear poisiton angles!
Basically: we have a good model for F814W but not F606W. Could try fitting the latter by itself...or keep soldiering on with these position angles and shear!!! With det8_alluniform.
X = 28 - emcee_2. This seems to have a way better logp, but I'm suspicious. Run for a short time to see what happens.

X = 0 - start of one colour run (F606W). det8_F606_shear_alluniform
X = 1 - as above, but with inflated pa covariances
'''


#this is with the bigger images. Have to be careful about adding things to the coordinates properly.
X = 1
# X = run count!
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
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.1,vmax=0.1,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-3, vmax = 3,cmap='afmhot',aspect='auto') #5,vmax=5,cmap='afmhot',aspect='auto')
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


guiFile = '/data/ljo31/Lens/J1347/basicmodel2'
#guiFile = '/data/ljo31/Lens/J1347/bm4'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2a'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2c'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2e'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2f'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2g'
guiFile = '/data/ljo31/Lens/J1347/basicmodel2h'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2h2'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2h3'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2h4'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2h4_alluniform'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2hi'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2h4_alluniform_srcpas'
#guiFile = '/data/ljo31/Lens/J1347/basicmodel2h7'
guiFile = '/data/ljo31/Lens/J1347/det8'
guiFile = '/data/ljo31/Lens/J1347/det8b'
guiFile = '/data/ljo31/Lens/J1347/det8_606_shear_alluniform'
#guiFile = '/data/ljo31/Lens/J1347/emcee_2'


print guiFile

img, sig, psf = img1.copy(), sig1.copy(), psf1.copy()

OVRS = 4
yc,xc = iT.overSample(img.shape,OVRS)
yc,xc = yc,xc
psf /= psf.sum()
psf = convolve.convolve(img,psf)[1]

G,L,S,offsets,_ = numpy.load(guiFile)

pars = []
cov = []
### first parameters need to be the offsets
#xoffset = offsets[0][3]
#yoffset = offsets[1][3]
#pars.append(pymc.Uniform('xoffset',-5.,5.,value=offsets[0][3]))
#pars.append(pymc.Uniform('yoffset',-5.,5.,value=offsets[1][3]))
#cov += [0.4,0.4]


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
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                if key == 'pa':
                    cov.append(s[key]['sdev']*100)
                else:
                    cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        for key in 'q','n','re','pa':
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
            else:
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=-0.01))
cov.append(0.02)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180.,value=130.))
cov.append(100.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

print gals, lenses, srcs

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = lensModel.lensFit(None,img, sig,gals,lenses,srcs,xc,yc,OVRS,
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
    S.sample(25*len(pars)**2)


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


im = lensModel.lensFit(coeff,img,sig,gals,lenses,srcs,xc,yc,OVRS,psf=psf,verbose=True) # return loglikelihood
im = lensModel.lensFit(coeff,img,sig,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True) # return model
model = lensModel.lensFit(coeff,img,sig,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    

NotPlicely(img,im,sig)


#numpy.save('/data/ljo31/Lens/J1606/trace'+str(Y), trace)
#numpy.save('/data/ljo31/Lens/J1606/logP'+str(Y), logp)

for key in det.keys():
    print key, '%.1f'%det[key][-1]
print 'max lnL is ', max(logp)


np.save('/data/ljo31/Lens/J1347/det_F606W_'+str(X),det)

#print 'x & y & n & re & q & pa \\'
print '&','&', '%.1f'%det['Source 1 n'][-1], '&', '%.1f'%det['Source 1 re'][-1], '&', '%.1f'%det['Source 1 q'][-1], '&', '%.1f'%det['Source 1 pa'][-1], '\\'
print ' &', '%.1f'%det['Source 2 n'][-1], '&', '%.1f'%det['Source 2 re'][-1], '&', '%.1f'%det['Source 2 q'][-1], '&', '%.1f'%det['Source 2 pa'][-1], '\\'
print '%.1f'%det['Galaxy 1 x'][-1], '&', '%.1f'%det['Galaxy 1 y'][-1], '&', '%.1f'%det['Galaxy 1 n'][-1], '&', '%.1f'%det['Galaxy 1 re'][-1], '&', '%.1f'%det['Galaxy 1 q'][-1], '&', '%.1f'%det['Galaxy 1 pa'][-1], '\\'
print '%.1f'%det['Lens 1 x'][-1], '&', '%.1f'%det['Lens 1 y'][-1], '&', '%.1f'%det['Lens 1 eta'][-1], '&', '%.1f'%det['Lens 1 b'][-1], '&', '%.1f'%det['Lens 1 q'][-1], '&', '%.1f'%det['Lens 1 pa'][-1], '\\'


