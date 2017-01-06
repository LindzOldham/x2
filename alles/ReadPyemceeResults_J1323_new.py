import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
import pylab as pl
import numpy as np
import lensModel2
from scipy.interpolate import RectBivariateSpline
from scipy import optimize

def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.3)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.3)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    #pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-3,vmax=3,cmap='afmhot',aspect='auto')
    #pl.colorbar()
    #pl.title('signal-to-noise residuals 1')
    pl.imshow((image-im),origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    #pl.colorbar()
    #pl.title('signal-to-noise residuals 1')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals 2')
    pl.colorbar()

result = cPickle.load(open('/data/ljo31/Lens/J1323/212_0'))
lp,trace,dic,_ = result
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0

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

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
mask = pyfits.open('/data/ljo31/Lens/J1323/mask2_F814W.fits')[0].data.copy()
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

xoffset,yoffset = dic['xoffset'][a1,a2,a3], dic['xoffset'][a1,a2,a3]

srcs,gals,lenses=[],[],[]
gals = []
for name in ['Galaxy 1','Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key] = gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

for name in ['Source 2','Source 1']:
    p = {}
    if name == 'Source 2':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Source 1':
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))

lenses = []
p = {}
for key in 'x','y','q','pa','b','eta':
    p[key] = dic['Lens 1 '+key][a1,a2,a3]
lenses.append(MassModels.PowerLaw('Lens 1',p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = dic['extShear'][a1,a2,a3]
p['pa'] = dic['extShear PA'][a1,a2,a3]
lenses.append(MassModels.ExtShear('shear',p))

colours = ['F606W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = xoffset
        dy = yoffset
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=23) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        print lens
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=23)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n] = 1*np.ones(model[n].shape)
    n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
    comps = False
    if comps == True:
        CotSomponents(components,colours[i])
    fits.append(fit)

print fits

pl.figure()
pl.plot(lp[:,0])


## also get parameters with errors!
### print table out with uncertainties
ftrace = trace[:,0].reshape((trace.shape[0]*trace.shape[2],trace.shape[3]))
upperlower = map(lambda v: (v[0],v[1],v[2]),zip(*np.percentile(ftrace,[16,50,84],axis=0)))
# this will change order depending on parameters, but I can't see a simple way to do this otherwise

d = []
l,u = [], []
for key in dic.keys():
    f = dic[key][:,0].reshape((trace.shape[0]*trace.shape[2]))
    lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
    d.append((key,med))
    l.append((key,lo))
    u.append((key,up))
    #print key, '$', '%.2f'%np.percentile(f,50), '_{-', '%.2f'%np.percentile(f,16), '}^{+', '%.2f'%np.percentile(f,84), '}$'

Ddic = dict(d)                    
Ldic = dict(l)
Udic = dict(u)
print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|c|cccccc|}\hline'
print r' object & x & y & re & n & pa & q \\\hline'
print 'source 1 & $', '%.2f'%(Ddic['Source 2 x']), '_{-', '%.2f'%Ldic['Source 2 x'],'}^{+','%.2f'%Udic['Source 2 x'], '}$ & $', '%.2f'%(Ddic['Source 2 y']),'_{-', '%.2f'%Ldic['Source 2 y'],'}^{+', '%.2f'%Udic['Source 2 y'], '}$ & $', '%.2f'%Ddic['Source 1 re'],'_{-', '%.2f'%Ldic['Source 1 re'],'}^{+', '%.2f'%Udic['Source 1 re'], '}$ & $', '%.2f'%Ddic['Source 1 n'],'_{-', '%.2f'%Ldic['Source 1 n'],'}^{+', '%.2f'%Udic['Source 1 n'], '}$ & $','%.2f'%Ddic['Source 1 pa'],'_{-', '%.2f'%Ldic['Source 1 pa'],'}^{+', '%.2f'%Udic['Source 1 pa'], '}$ & $','%.2f'%Ddic['Source 1 q'],'_{-', '%.2f'%Ldic['Source 1 q'],'}^{+', '%.2f'%Udic['Source 1 q'], '}$',r'\\'
###
print 'source 2 & $', '%.2f'%(Ddic['Source 2 x']), '_{-', '%.2f'%Ldic['Source 2 x'],'}^{+','%.2f'%Udic['Source 2 x'], '}$ & $', '%.2f'%(Ddic['Source 2 y']),'_{-', '%.2f'%Ldic['Source 2 y'],'}^{+', '%.2f'%Udic['Source 2 y'], '}$ & $', '%.2f'%Ddic['Source 2 re'],'_{-', '%.2f'%Ldic['Source 2 re'],'}^{+', '%.2f'%Udic['Source 2 re'], '}$ & $', '%.2f'%Ddic['Source 2 n'],'_{-', '%.2f'%Ldic['Source 2 n'],'}^{+', '%.2f'%Udic['Source 2 n'], '}$ & $','%.2f'%Ddic['Source 2 pa'],'_{-', '%.2f'%Ldic['Source 2 pa'],'}^{+', '%.2f'%Udic['Source 2 pa'], '}$ & $','%.2f'%Ddic['Source 2 q'],'_{-', '%.2f'%Ldic['Source 2 q'],'}^{+', '%.2f'%Udic['Source 2 q'], '}$',r'\\'


print 'galaxy 1 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 1 re'],'_{-', '%.2f'%Ldic['Galaxy 1 re'],'}^{+', '%.2f'%Udic['Galaxy 1 re'], '}$ & $', '%.2f'%Ddic['Galaxy 1 n'],'_{-', '%.2f'%Ldic['Galaxy 1 n'],'}^{+', '%.2f'%Udic['Galaxy 1 n'], '}$ & $','%.2f'%Ddic['Galaxy 1 pa'],'_{-', '%.2f'%Ldic['Galaxy 1 pa'],'}^{+', '%.2f'%Udic['Galaxy 1 pa'], '}$ & $','%.2f'%Ddic['Galaxy 1 q'],'_{-', '%.2f'%Ldic['Galaxy 1 q'],'}^{+', '%.2f'%Udic['Galaxy 1 q'], '}$',r'\\'
###
print 'galaxy 2 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 2 re'],'_{-', '%.2f'%Ldic['Galaxy 2 re'],'}^{+', '%.2f'%Udic['Galaxy 2 re'], '}$ & $', '%.2f'%Ddic['Galaxy 2 n'],'_{-', '%.2f'%Ldic['Galaxy 2 n'],'}^{+', '%.2f'%Udic['Galaxy 2 n'], '}$ & $','%.2f'%Ddic['Galaxy 2 pa'],'_{-', '%.2f'%Ldic['Galaxy 2 pa'],'}^{+', '%.2f'%Udic['Galaxy 2 pa'], '}$ & $','%.2f'%Ddic['Galaxy 2 q'],'_{-', '%.2f'%Ldic['Galaxy 2 q'],'}^{+', '%.2f'%Udic['Galaxy 2 q'], '}$',r'\\'
###
print 'lens 1 & $', '%.2f'%Ddic['Lens 1 x'], '_{-', '%.2f'%Ldic['Lens 1 x'],'}^{+','%.2f'%Udic['Lens 1 x'], '}$ & $', '%.2f'%Ddic['Lens 1 y'],'_{-', '%.2f'%Ldic['Lens 1 y'],'}^{+', '%.2f'%Udic['Lens 1 y'], '}$ & $', '%.2f'%Ddic['Lens 1 b'],'_{-', '%.2f'%Ldic['Lens 1 b'],'}^{+', '%.2f'%Udic['Lens 1 b'], '}$ & $', '%.2f'%Ddic['Lens 1 eta'],'_{-', '%.2f'%Ldic['Lens 1 eta'],'}^{+', '%.2f'%Udic['Lens 1 eta'], '}$ & $','%.2f'%Ddic['Lens 1 pa'],'_{-', '%.2f'%Ldic['Lens 1 pa'],'}^{+', '%.2f'%Udic['Lens 1 pa'], '}$ & $','%.2f'%Ddic['Lens 1 q'],'_{-', '%.2f'%Ldic['Lens 1 q'],'}^{+', '%.2f'%Udic['Lens 1 q'], '}$',r'\\\hline'
###
print r'\end{tabular}'
print r'\caption{', 'shear = $', '%.2f'%Ddic['extShear'], '_{-', '%.2f'%Ldic['extShear'],'}^{+','%.2f'%Udic['extShear'], '}$ , shear pa = $',  '%.2f'%Ddic['extShear PA'], '_{-', '%.2f'%Ldic['extShear PA'],'}^{+','%.2f'%Udic['extShear PA'], '}$}'
print r'\end{table}'
