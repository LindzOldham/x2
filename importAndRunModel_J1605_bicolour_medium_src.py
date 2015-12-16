import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np

'''
no X = BasicFit10 - with two galaxy components
X = 2 - BM2. This has one galaxy component. Also reduced the cutout size of psf1 and renormalised - this should reduce the noise introduced by the convolution. 
X = 3 - terminal_iterated. This is the output from X=0 (ie. no X), 
X = 4 - terminal_iterated_2. This has source 1 pa = 130, q = 0.7
X = 5 - terminak)_iterated_3
X = 6 - terminal_iterated but run for longer!
X = 7 - terminal_iterated_2 for 80 * p**2
X = 8 - terminal_iterated_3 for 80  * p**2!
X = 9 - terminal_iterated_4 - we've started playing around with src2 now.
X = 10 - terminal_iterated_4, but with sources and galaxies both fixed to lie on top of each other!
X = 'FINAL' - what it says mate.
X = 11 - removing one source component and comparing the fit. Aim is to show that two components are necessary
'''

#this is with the bigger images. Have to be careful about adding things to the coordinates properly.
X = 11
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
    #pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto') #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    #pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25,cmap='afmhot',aspect='auto')
    pl.colorbar()
    #pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    #pl.title('signal-to-noise residuals')
    pl.colorbar()
    pl.subplots_adjust(left=0.05,bottom=0.05,top=0.92,right=0.95)
    pl.subplots_adjust(wspace=0,hspace=0.1)
    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')


img1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2_masked.fits')[0].data.copy() # masking out the possiblyasecondsource regions!
psf1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf2= psf2[15:-16,14:-16]
psf2 /= psf2.sum()

#guiFile = '/data/ljo31/Lens/J1605/fit3'
guiFile = '/data/ljo31/Lens/J1605/fit4'
#guiFile = '/data/ljo31/Lens/J1605/fit9'
guiFile = '/data/ljo31/Lens/J1605/fit10b'
guiFile = '/data/ljo31/Lens/J1605/fit11'
guiFile = '/data/ljo31/Lens/J1605/BasicModel10'
guiFile = '/data/ljo31/Lens/J1605/BasicModel10b'
guiFile = '/data/ljo31/Lens/J1605/BasicModel10c'
guiFile = '/data/ljo31/Lens/J1605/BasicFit10d'
guiFile = '/data/ljo31/Lens/J1605/BM2'
guiFile = '/data/ljo31/Lens/J1605/terminal_bestfit_iterated'
#guiFile = '/data/ljo31/Lens/J1605/terminal_bestfit_iterated_2'
#guiFile = '/data/ljo31/Lens/J1605/terminal_bestfit_iterated_3'
guiFile = '/data/ljo31/Lens/J1605/terminal_iterated_4'
#guiFile = '/data/ljo31/Lens/J1605/SingleSource'


print 'schon aus Terminal'

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
yc,xc = iT.overSample(img1.shape,1.)
yc,xc = yc-15.,xc-15.
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
print xoffset,yoffset
print offsets
pars.append(pymc.Uniform('xoffset',-5.,5.,value=offsets[0][3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=offsets[1][3]))
cov += [0.4,0.4]

srcs = []
for name in 'Source 1', 'Source 2':
    s = S[name]
    p = {}
    if name == 'Source 1':
        for key in 'x','y','q','pa','re','n':
            if s[key]['type']=='constant':
                p[key] = s[key]['value']
            else:
                lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
                pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
                p[key] = pars[-1]
                cov.append(s[key]['sdev'])
    elif name == 'Source 2':
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
for name in 'Galaxy 1', 'Galaxy 2':
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
        print i
        if i == 0:
            x0,y0 = 0,0
        else:
            x0 = pars[0].value 
            y0 = pars[1].value 
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

plot = True
if plot:
    for i in range(len(keylist)):
        pl.figure()
        pl.plot(chainlist[i])
        pl.title(str(keylist[i]))


pl.figure()
pl.plot(logp)
pl.title('log P')

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
    print im
    im = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True) # return model
    model = lensModel.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    ims.append(im)
    models.append(model)

colours = ['F555W', 'F814W']
for i in range(len(imgs)):
    image = imgs[i]
    im = ims[i]
    model = models[i]
    sigma = sigs[i]
    #pyfits.PrimaryHDU(model).writeto('/data/ljo31/Lens/J1605/components_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #pyfits.PrimaryHDU(im).writeto('/data/ljo31/Lens/J1605/model_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #pyfits.PrimaryHDU(image-im).writeto('/data/ljo31/Lens/J1605/resid_uniform'+str(colours[i])+str(X)+'.fits',clobber=True)
    #f = open('/data/ljo31/Lens/J1605/coeff'+str(X),'wb')
    #cPickle.dump(coeff,f,2)
    #f.close()
    NotPlicely(image,im,sigma)
    pl.suptitle(str(colours[i]))


### OUTPUT THE THINGS IN LATEX-FRIENDLY FORM!
print '%.1f'%det['Source 1 x'][-1], '&', '%.1f'%det['Source 1 y'][-1], '&', '%.1f'%det['Source 1 n'][-1], '&', '%.1f'%det['Source 1 re'][-1], '&', '%.1f'%det['Source 1 q'][-1], '&', '%.1f'%det['Source 1 pa'][-1], '\\'
#
print '%.1f'%det['Source 2 n'][-1], '&', '%.1f'%det['Source 2 re'][-1], '&', '%.1f'%det['Source 2 q'][-1], '&', '%.1f'%det['Source 2 pa'][-1], '\\'

numpy.save('/data/ljo31/Lens/J1605/trace'+str(X), trace)
numpy.save('/data/ljo31/Lens/J1605/logP'+str(X), logp)

for key in det.keys():
    print key, '%.1f'%det[key][-1]
print 'max lnL is ', max(logp)

#pl.figure()
#pl.imshow((im-image)/sigma)
#pl.colorbar()
print det['xoffset'], det['yoffset']
#print xoffset, yoffset

np.save('/data/ljo31/Lens/J1605/det'+str(X),det)

'''
## radial gradients?
onmodels = []
import lensModel2
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = det['xoffset'][-1], det['yoffset'][-1] # xoffset, yoffset #
        print x0,y0
    model = lensModel2.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    onmodels.append(model)
'''
'''
## fits:
gal1, gal2, src1, src2
gal1 - small
gal2 - big
src1 - small
src2 - big
GET (eg. for the iterated terminal model)
                gal1 (small), gal2 (big), src1 (small), src2 (big)
fit_F555W =  [ 0.02280558  0.00560575  0.00101855  0.01002717]

fit_F814W =  [ 0.03723596  0.03389582  0.00116828  0.13499762]


In the F555W image, the small galaxy component is dominant, whereas in the F814W image, both large and small are pretty equal. Both images have a similar contribution from the small source, and both are dominated by the big source contribution. However, this is about 10 times bigger in the F814W band...!


'''

## now we can extract the results and construct the source galaxy in the source plane!
srcs = []
p1,p2 = {},{}
for name in det.keys():
    s = det[name]
    if name[:8] == 'Source 1':
        for key in 'x','y','q','pa','re','n':
            p1[key] = s[-1]
    elif name[:8] == 'Source 2':
        for key in 'x','y','q','pa','re','n':
            p2[key] = s[-1]
               
srcs.append(SBModels.Sersic('Source 1',p1))
srcs.append(SBModels.Sersic('Source 2',p2))
ims = []
tims = np.zeros(imgs[0].shape)
for i in range(len(srcs)):
    src = srcs[i]
    im = src.pixeval(xc,yc)
    ims.append(im)
    tims +=im

pl.figure()
pl.imshow(tims,origin-'lower',interpolation='nearest')
for i in range(2):
    pl.figure()
    pl.imshow(ims[i],origin='lower',interpolation='nearest')


### SOURCE
# physical scale: z_src = 0.542 so 6.432 kpc/arcsec
# image scale: 0.05 arcsec/pixel
# so: 0.32 kpc/pixel

### GALAXY
# z_gal = 0.306 so 4.556 kpc/arcsec
# 0.2278 kpc /pixel


## to load up a det and plot it
img1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits')[0].data.copy()
sig1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_noisemap2_masked.fits')[0].data.copy() # masking out the possiblyasecondsource regions!
psf1 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits')[0].data.copy()
psf1 = psf1[10:-10,10:-10]
psf1 = psf1/np.sum(psf1)

img2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf2= psf2[15:-16,14:-16]
psf2 /= psf2.sum()

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

OVRS=1.
det = np.load('/data/ljo31/Lens/J1605/det10.npy')[()] # this has the galaxies and sources coincident in space.
srcs = []
gals = []
lenses = []
coeff=[]
g1,g2,l1,s1,s2,sh = {},{},{},{},{},{}
for name in det.keys():
    s = det[name]
    coeff.append(s[-1])
    if name[:8] == 'Source 1':
        s1[name[9:]] = s[-1]
    elif name[:8] == 'Source 2':
        s2[name[9:]] = s[-1]
    elif name[:6] == 'Lens 1':
        l1[name[7:]] = s[-1]
    elif name[:8] == 'Galaxy 1':
        g1[name[9:]] = s[-1]
    elif name[:8] == 'Galaxy 2':
        g2[name[9:]] = s[-1]
    elif name[:8] == 'extShear':
        if len(name)<9:
            sh['b'] = s[-1]
        elif name == 'extShear PA':
            sh['pa'] = s[-1]
    
s2['x'] = s1['x'].copy()
s2['y'] = s1['y'].copy()
g2['x'] = g1['x'].copy()
g2['y'] = g1['y'].copy()
srcs.append(SBModels.Sersic('Source 1',s1))
srcs.append(SBModels.Sersic('Source 2',s2))
gals.append(SBModels.Sersic('Galaxy 1',g1))
gals.append(SBModels.Sersic('Galaxy 2',g2))
lenses.append(MassModels.PowerLaw('Lens 1',l1))
sh['x'] = lenses[0].pars['x']
sh['y'] = lenses[0].pars['y']
lenses.append(MassModels.ExtShear('shear',sh))


import lensModel2
ims = []
models = []
sfit = []
for i in range(len(imgs)):
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    if i == 0:
        x0,y0 = 0,0
    else:
        x0,y0 = det['xoffset'][-1], det['yoffset'][-1] # xoffset, yoffset #
        print x0,y0
    model = lensModel2.lensFit(coeff,image,sigma,gals,lenses,srcs,xc+x0,yc+y0,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True) # return the model decomposed into the separate galaxy and source components
    sfit.append([model[2],model[3]])

ims = []
tims = np.zeros(imgs[0].shape)
for i in range(len(srcs)):
    src = srcs[i]
    im = src.pixeval(xc,yc) * sfit[0][i]
    ims.append(im)
    tims +=im
    pl.figure()
    pl.imshow(im,origin='lower',interpolation='nearest')
    pl.colorbar()


pl.figure()
pl.imshow(tims,origin='lower',interpolation='nearest')

