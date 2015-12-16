import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
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
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25)
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5)
    pl.title('signal-to-noise residuals')
    pl.colorbar()


img2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits')[0].data.copy()
sig2 = pyfits.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_noisemap2_masked.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/J1605/F814W_psf_#2.fits')[0].data.copy()  
psf2= psf2[15:-16,14:-16]
psf2 /= psf2.sum()
psf = convolve.convolve(img2,psf2)[1]
OVRS=1.


fchain = np.load('/data/ljo31/Lens/J1605/fchain_PixFit_2src.npy')
flnprob = np.load('/data/ljo31/Lens/J1605/flnprob_PixFit_2src.npy')
fit = np.load('/data/ljo31/Lens/J1605/fit_PixFit_2src.npy')
ii = np.argmax(flnprob)
XX = fchain[ii]
x,y,pa,q,re,n,pa2,q2,re2,n2 = fchain[ii]
srcs = []
src1 = SBModels.Sersic('Source 1', {'x':0.1196461*x+10.91,'y':0.11966*y + 5.877,'pa':pa,'q':q,'re':re*0.1196461,'n':n,'amp':fit[0]})
src2 = SBModels.Sersic('Source 2', {'x':0.1196461*x+10.91,'y':0.11966*y + 5.877,'pa':pa2,'q':q2,'re':re2*0.1196461,'n':n2,'amp':fit[1]})
srcs = [src1,src2]


guiFile = '/data/ljo31/Lens/J1605/terminal_iterated_4'
G,L,S,offsets,shear = numpy.load(guiFile)
x0 = offsets[0][3]
y0 = offsets[1][3]
yc,xc = iT.coords(img2.shape)-15
yc,xc = yc+y0,xc+x0

cov = []
pars = []
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


im = lensModel.lensFit(None,img2,sig2,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True)
#NotPlicely(img2,im,sig2)

##
mask = pyfits.open('/data/ljo31/Lens/J1605/maskF814W.fits')[0].data.copy()
mask = mask==1
xm = xc[mask]
ym = yc[mask]
coords = [xm,ym]
xl,yl = pylens.getDeflections(lenses,coords)
span = max(xl.max()-xl.min(),yl.max()-yl.min())
npix=400.
xmin,xmax,ymin,ymax = xl.mean()-span/2.,xl.mean()+span/2.,yl.mean()-span/2.,yl.mean()+span/2.
scalex,scaley = (xmax-xmin)/npix,(ymax-ymin)/npix
xn,yn = xmin,ymin
print scalex,scaley,xn,yn

x,y,pa,q,re,n,pa2,q2,re2,n2 = fchain[ii]
srcs = []
src1 = SBModels.Sersic('Source 1', {'x':scalex*x+xn,'y':scaley*y+yn,'pa':pa,'q':q,'re':re*0.1196461,'n':n,'amp':fit[0]})
src2 = SBModels.Sersic('Source 2', {'x':scalex*x+xn,'y':scaley*y+yn,'pa':pa2,'q':q2,'re':re2*0.1196461,'n':n2,'amp':fit[1]})
srcs = [src1,src2]

im = lensModel.lensFit(None,img2,sig2,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True)
#NotPlicely(img2,im,sig2)


###

#pars = []
pars.append(pymc.Uniform('xoffset',-5.,5.,value=0.1))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=0.1))
cov += [0.4,0.4]
optCov = np.array(cov)

# fit for offsets
@pymc.deterministic
def logP(value=0.,p=pars):
    x0 = pars[-2].value
    y0 = pars[-1].value
    x,y,pa,q,re,n,pa2,q2,re2,n2 = XX
    src1 = SBModels.Sersic('Source 1', {'x':0.1196461*x+10.91+x0,'y':0.11966*y + 5.877+y0,'pa':pa,'q':q,'re':re*0.1196461,'n':n,'amp':fit[0]})
    src2 = SBModels.Sersic('Source 2', {'x':0.1196461*x+10.91+x0,'y':0.11966*y + 5.877+y0,'pa':pa2,'q':q2,'re':re2*0.1196461,'n':n2,'amp':fit[1]})
    srcs = [src1,src2]
    return lensModel.lensFit(None,img2,sig2,gals,lenses,srcs,xc,yc,1,verbose=False,psf=psf,csub=1)
   

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp


for i in range(1):
    S = AMAOpt(pars,[likelihood],[logP],cov=optCov/4.)
    S.set_minprop(len(pars)*2)
    S.sample(5*len(pars)**2)

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

x0,y0 = det['xoffset'][-1], det['yoffset'][-1]
x,y,pa,q,re,n,pa2,q2,re2,n2 = XX
src1 = SBModels.Sersic('Source 1', {'x':0.1196461*x+10.91+x0,'y':0.11966*y + 5.877+y0,'pa':pa,'q':q,'re':re*0.1196461,'n':n,'amp':fit[0]})
src2 = SBModels.Sersic('Source 2', {'x':0.1196461*x+10.91+x0,'y':0.11966*y + 5.877+y0,'pa':pa2,'q':q2,'re':re2*0.1196461,'n':n2,'amp':fit[1]})
srcs = [src1,src2]
im = lensModel.lensFit(None,img2,sig2,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True)
NotPlicely(img2,im,sig2)

'''
## let's also compare our two models? Pixellated/analytic and direct-analytic
x,y,pa,q,re,n,pa2,q2,re2,n2 = XX
src1 = SBModels.Sersic('Source 1', {'x':0.1196461*x+10.91+x0,'y':0.11966*y + 5.877+y0,'pa':pa,'q':q,'re':re*0.1196461,'n':n,'amp':fit[0]})
src2 = SBModels.Sersic('Source 2', {'x':0.1196461*x+10.91+x0,'y':0.11966*y + 5.877+y0,'pa':pa2,'q':q2,'re':re2*0.1196461,'n':n2,'amp':fit[1]})
pix = src1.pixeval(xc,yc) + src2.pixeval(xc,yc)
pl.figure()
pl.imshow(pix,interpolation='nearest',origin='lower',vmin=0,vmax=5)

pl.figure()
x2 = 0.1196461*x+10.91+x0
y2 = 0.11966*y + 5.877+y0
R2 = np.sqrt((xc-x2)**2. + (yc-y2)**2.)
#pl.scatter(np.ravel(R2), np.ravel(pix))
R2 = np.ravel(R2)
sort = R2.argsort()
pl.plot(R2[sort],np.ravel(pix)[sort])

G,L,S,offsets,shear = numpy.load(guiFile)
srcs = []
for name in 'Source 1', 'Source 2':
    s = S[name]
    p = {}
    if name == 'Source 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = s[key]['value']
    elif name == 'Source 2':
        for key in 'q','pa','re','n':
            p[key] = s[key]['value']
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))

import lensModel2
fit = lensModel2.lensFit(coeff,img2,sig2,gals,lenses,srcs,xc,yc,OVRS,noResid=True,psf=psf,verbose=True,getModel=True,showAmps=True)
print fit
print fit.shape

sources = srcs[0].pixeval(xc,yc)*fit[2] + srcs[1].pixeval(xc,yc)*fit[3]
pl.figure()
pl.imshow(sources,interpolation='nearest',origin='lower',vmin=0,vmax=5)
x1,y1 = srcs[0].pars['x'], srcs[0].pars['y']
q1 = srcs[0]
R = np.sqrt((xc-x1)**2. + (yc-y1)**2.)
pl.figure()
R = np.ravel(R)
sort2 = R.argsort()
pl.plot(R[sort2], np.ravel(sources)[sort])

pl.figure()
R = np.sqrt((xc-x1)**2. + (yc-y1)**2.)
'''
