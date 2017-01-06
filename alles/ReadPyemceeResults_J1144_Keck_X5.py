import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
#import myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline


''' This code now also calculates the source position relative to the lens rather than relative to the origin. This means that when the lens moves, the source moves with it! I have tested this in so far as it seems to produce the same results on the final inference as before. Should maybe test it on an earlier model incarnation though.'''

# plot things
def NotPlicely(image,im,sigma,colour):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=8)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=8)
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
    py.writeto('/data/ljo31/Lens/J1144/resid'+str(colour)+'.fits',(image-im)/sigma,clobber=True)
    py.writeto('/data/ljo31/Lens/J1144/model'+str(colour)+'.fits',im,clobber=True)


image = py.open('/data/ljo31/Lens/J1144/J1144_Kp_narrow_cutout.fits')[0].data.copy()
sigma = np.ones(image.shape)
guiFile = '/data/ljo31/Lens/J1144/FINAL_1src_30'
G,L,S,offsets,shear = numpy.load(guiFile)
print guiFile

result = np.load('/data/ljo31/Lens/J1144/KeckPSF_5')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

OVRS = 1
yc,xc = iT.overSample(image.shape,OVRS)
yo,xo = iT.overSample(image.shape,1)
xc,xo,yc,yo=xc*0.2,xo*0.2,yc*0.2,yo*0.2
xc,xo = xc+12 , xo+12 
yc,yo = yc+20 , yo+20 
mask = np.zeros(image.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0
xpsf,ypsf = iT.coords((340,340))-170

dx,dy,sig1,q1,pa1,amp1,sig2,q2,pa2 = trace[a1,a2,a3]
amp2=1.-amp1
gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] =s[key]['value']
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            p[key] =s[key]['value']
        for key in 'x','y':
            p[key]=gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))


lenses = []
for name in L.keys():
    s = L[name]
    p = {}
    for key in 'x','y','q','pa','b','eta':
        p[key] =s[key]['value']
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = shear[0]['b']['value']
p['pa'] = shear[0]['pa']['value']
lenses.append(MassModels.ExtShear('shear',p))


srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
            p[key] =s[key]['value']
        for key in 'x','y':
            p[key] = s[key]['value']
    srcs.append(SBModels.Sersic(name,p))
psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':sig1,'q':q1,'pa':pa1,'amp':10})
psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':sig2,'q':q2,'pa':pa2,'amp':10})
psf1 = psfObj1.pixeval(xpsf,ypsf) * amp1 / (np.pi*2.*sig1**2.)
psf2 = psfObj2.pixeval(xpsf,ypsf) * amp2 / (np.pi*2.*sig2**2.)
psf = psf1 + psf2 
psf /= psf.sum()
psf = convolve.convolve(image,psf)[1]
xp,yp = xc+dx,yc+dy
imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
n = 0
model = np.empty(((len(gals) + len(srcs)+1),imin.size))
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
model[n] = np.ones(model[n-1].shape)
n+=1
rhs = (imin/sigin) # data
op = (model/sigin).T # model matrix
fit, chi = optimize.nnls(op,rhs)
components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
model = components.sum(0)
NotPlicely(image,model,sigma,'Kp')
#pl.suptitle('Kp')
    
pl.figure()
pl.imshow(psf1,interpolation='nearest')
pl.colorbar()
pl.figure()
pl.imshow(psf2,interpolation='nearest')
pl.colorbar()
pl.figure()
pl.imshow(psf1+psf2,interpolation='nearest')
pl.colorbar()

print '%.2f'%dx,'%.2f'%dy,'%.2f'%sig1,'%.2f'%q1,'%.2f'%pa1,'%.2f'%amp1,'%.2f'%sig2,'%.2f'%q2,'%.2f'%pa2
