import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles
'''
X=0 - TO RUN
'''

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0)#,vmax=560) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0)#,vmax=560) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()

image = py.open('/data/ljo31/Lens/J1446/EEL1446_med.fits')[0].data.copy()[730:895,630:920]
sigma = np.ones(image.shape) 
psf2 = py.open('/data/ljo31/Lens/J0837/PSF_nirc2_Kp_narrow.fits')[0].data.copy()[25:-25,25:-25]
psf2 = psf2/np.sum(psf2)


img,sig=image.copy(),sigma.copy()
result = np.load('/data/ljo31/Lens/J1446/KeckPSF_1')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

OVRS = 1
yc,xc = iT.overSample(image.shape,OVRS)
yo,xo = iT.overSample(image.shape,1)
#xc,xo,yc,yo = xc-60,xo-60,yc-40,yo-40

#xc,xo,yc,yo = xc+250,xo+250,yc+200,yo+200
xc,xo,yc,yo=xc*0.2,xo*0.2,yc*0.2,yo*0.2
#xc,xo,yc,yo = xc-140,xo-140,yc-120,yo-120
xc,xo=xc+21,xo+21
yc,yo=yc+14,yo+14
mask = np.zeros(image.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0


imgs = [img]
sigs = [sig]
psf = psf2
psf /= psf.sum()
psf = convolve.convolve(image,psf)[1]
PSFs = [psf]

### first parameters need to be the offsets
xoffset=dic['xoffset'][a1,a2,a3]
yoffset=dic['yoffset'][a1,a2,a3]

gals = []
for name in ['Galaxy 1','Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    gals.append(SBModels.Sersic(name,p))

lenses = []
for name in ['Lens 1']:
    p = {}
    for key in 'x','y','q','pa','b','eta':
        p[key] = p[key] = dic[name+' '+key][a1,a2,a3]
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
p['b'] = dic['extShear'][a1,a2,a3]
p['pa'] = dic['extShear PA'][a1,a2,a3]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in ['Source 2','Source 1']:
    p = {}
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': 
           p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))




models = []
for i in range(len(imgs)):
    dx = xoffset 
    dy = yoffset 
    xp,yp = xc+dx,yc+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=11)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    model[n] = np.ones(model[n-1].size)
    n+=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    NotPlicely(image,model,sigma)
    for i in range(4):
        pl.figure()
        pl.imshow(components[i],interpolation='nearest',origin='lower')
        pl.colorbar()
    print fit

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
print 'source 1 & $', '%.2f'%(Ddic['Source 2 x']+Ddic['Lens 1 x']), '_{-', '%.2f'%Ldic['Source 2 x'],'}^{+','%.2f'%Udic['Source 2 x'], '}$ & $', '%.2f'%(Ddic['Source 2 y']+Ddic['Lens 1 y']),'_{-', '%.2f'%Ldic['Source 2 y'],'}^{+', '%.2f'%Udic['Source 2 y'], '}$ & $', '%.2f'%Ddic['Source 1 re'],'_{-', '%.2f'%Ldic['Source 1 re'],'}^{+', '%.2f'%Udic['Source 1 re'], '}$ & $', '%.2f'%Ddic['Source 1 n'],'_{-', '%.2f'%Ldic['Source 1 n'],'}^{+', '%.2f'%Udic['Source 1 n'], '}$ & $','%.2f'%Ddic['Source 1 pa'],'_{-', '%.2f'%Ldic['Source 1 pa'],'}^{+', '%.2f'%Udic['Source 1 pa'], '}$ & $','%.2f'%Ddic['Source 1 q'],'_{-', '%.2f'%Ldic['Source 1 q'],'}^{+', '%.2f'%Udic['Source 1 q'], '}$',r'\\'
###
print 'source 2 & $', '%.2f'%(Ddic['Source 2 x']+Ddic['Lens 1 x']), '_{-', '%.2f'%Ldic['Source 2 x'],'}^{+','%.2f'%Udic['Source 2 x'], '}$ & $', '%.2f'%(Ddic['Source 2 y']+Ddic['Lens 1 y']),'_{-', '%.2f'%Ldic['Source 2 y'],'}^{+', '%.2f'%Udic['Source 2 y'], '}$ & $', '%.2f'%Ddic['Source 2 re'],'_{-', '%.2f'%Ldic['Source 2 re'],'}^{+', '%.2f'%Udic['Source 2 re'], '}$ & $', '%.2f'%Ddic['Source 2 n'],'_{-', '%.2f'%Ldic['Source 2 n'],'}^{+', '%.2f'%Udic['Source 2 n'], '}$ & $','%.2f'%Ddic['Source 2 pa'],'_{-', '%.2f'%Ldic['Source 2 pa'],'}^{+', '%.2f'%Udic['Source 2 pa'], '}$ & $','%.2f'%Ddic['Source 2 q'],'_{-', '%.2f'%Ldic['Source 2 q'],'}^{+', '%.2f'%Udic['Source 2 q'], '}$',r'\\'
###

print 'galaxy 1 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 1 re'],'_{-', '%.2f'%Ldic['Galaxy 1 re'],'}^{+', '%.2f'%Udic['Galaxy 1 re'], '}$ & $', '%.2f'%Ddic['Galaxy 1 n'],'_{-', '%.2f'%Ldic['Galaxy 1 n'],'}^{+', '%.2f'%Udic['Galaxy 1 n'], '}$ & $','%.2f'%Ddic['Galaxy 1 pa'],'_{-', '%.2f'%Ldic['Galaxy 1 pa'],'}^{+', '%.2f'%Udic['Galaxy 1 pa'], '}$ & $','%.2f'%Ddic['Galaxy 1 q'],'_{-', '%.2f'%Ldic['Galaxy 1 q'],'}^{+', '%.2f'%Udic['Galaxy 1 q'], '}$',r'\\'
###
print 'galaxy 2 & $', '%.2f'%Ddic['Galaxy 2 x'], '_{-', '%.2f'%Ldic['Galaxy 2 x'],'}^{+','%.2f'%Udic['Galaxy 2 x'], '}$ & $', '%.2f'%Ddic['Galaxy 2 y'],'_{-', '%.2f'%Ldic['Galaxy 2 y'],'}^{+', '%.2f'%Udic['Galaxy 2 y'], '}$ & $', '%.2f'%Ddic['Galaxy 2 re'],'_{-', '%.2f'%Ldic['Galaxy 2 re'],'}^{+', '%.2f'%Udic['Galaxy 2 re'], '}$ & $', '%.2f'%Ddic['Galaxy 2 n'],'_{-', '%.2f'%Ldic['Galaxy 2 n'],'}^{+', '%.2f'%Udic['Galaxy 2 n'], '}$ & $','%.2f'%Ddic['Galaxy 2 pa'],'_{-', '%.2f'%Ldic['Galaxy 2 pa'],'}^{+', '%.2f'%Udic['Galaxy 2 pa'], '}$ & $','%.2f'%Ddic['Galaxy 2 q'],'_{-', '%.2f'%Ldic['Galaxy 2 q'],'}^{+', '%.2f'%Udic['Galaxy 2 q'], '}$',r'\\'
###
print 'lens 1 & $', '%.2f'%Ddic['Lens 1 x'], '_{-', '%.2f'%Ldic['Lens 1 x'],'}^{+','%.2f'%Udic['Lens 1 x'], '}$ & $', '%.2f'%Ddic['Lens 1 y'],'_{-', '%.2f'%Ldic['Lens 1 y'],'}^{+', '%.2f'%Udic['Lens 1 y'], '}$ & $', '%.2f'%Ddic['Lens 1 b'],'_{-', '%.2f'%Ldic['Lens 1 b'],'}^{+', '%.2f'%Udic['Lens 1 b'], '}$ & $', '%.2f'%Ddic['Lens 1 eta'],'_{-', '%.2f'%Ldic['Lens 1 eta'],'}^{+', '%.2f'%Udic['Lens 1 eta'], '}$ & $','%.2f'%Ddic['Lens 1 pa'],'_{-', '%.2f'%Ldic['Lens 1 pa'],'}^{+', '%.2f'%Udic['Lens 1 pa'], '}$ & $','%.2f'%Ddic['Lens 1 q'],'_{-', '%.2f'%Ldic['Lens 1 q'],'}^{+', '%.2f'%Udic['Lens 1 q'], '}$',r'\\\hline'
###
print r'\end{tabular}'
print r'\caption{', 'shear = $', '%.2f'%Ddic['extShear'], '_{-', '%.2f'%Ldic['extShear'],'}^{+','%.2f'%Udic['extShear'], '}$ , shear pa = $',  '%.2f'%Ddic['extShear PA'], '_{-', '%.2f'%Ldic['extShear PA'],'}^{+','%.2f'%Udic['extShear PA'], '}$}'
print r'\end{table}'
