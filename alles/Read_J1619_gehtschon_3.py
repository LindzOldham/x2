import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
#import myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
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

img1 = py.open('/data/ljo31/Lens/J1619/F606W_sci_cutout.fits')[0].data.copy()[30:-38,25:-30]
sig1 = py.open('/data/ljo31/Lens/J1619/F606W_noisemap.fits')[0].data.copy()[30:-38,25:-30]
psf1 = py.open('/data/ljo31/Lens/J1619/F606W_psf4.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)
psf3 = psf1.copy()
psf3[5:-5,5:-5] = np.nan
cond = np.isfinite(psf3)
m = psf3[cond].mean()
psf1 = psf1 - m
psf1 = psf1/np.sum(psf1)


img2 = py.open('/data/ljo31/Lens/J1619/F814W_sci_cutout.fits')[0].data.copy()[30:-38,25:-30]
sig2 = py.open('/data/ljo31/Lens/J1619/F814W_noisemap.fits')[0].data.copy()[30:-38,25:-30] 
psf2 = py.open('/data/ljo31/Lens/J1619/F814W_psf1neu.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)
psf3 = psf2.copy()
psf3[5:-5,5:-5] = np.nan
cond = np.isfinite(psf3)
m = psf3[cond].mean()
psf2 = psf2 - m
psf2 = psf2/np.sum(psf2)

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
print np.mean(xo), np.mean(yo)

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

result = np.load('/data/ljo31/Lens/J1619/gehtschon_10')
lp,trace,dic,_ = result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
xoffset,yoffset = dic['xoffset'][a1,a2,a3], dic['yoffset'][a1,a2,a3]

gals = []
for name in ['Galaxy 1']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key]=gals[0].pars[key]
    gals.append(SBModels.Sersic(name,p))

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
#p = {}
#p['x'] = dic['blob x'][a1,a2,a3]
#p['y'] = dic['blob y'][a1,a2,a3]
#p['b'] = dic['blob b'][a1,a2,a3]
#p['eta'] = dic['blob eta'][a1,a2,a3]
#p['pa'],p['q'] = 0.,0.9
#lenses.append(MassModels.PowerLaw('blob',p))

srcs = []
for name in ['Source 1','Source 2']:
    p = {}
    if name == 'Source 2':
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': 
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
    elif name == 'Source 1':
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': 
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))


npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]


for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = xoffset 
        dy = yoffset 
    xp,yp = xc+dx,yc+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xp*0.
        tmp = gal.pixeval(xin,yin,1./OVRS,csub=23).reshape(xp.shape)
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xp*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=23).reshape(xp.shape)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n] = np.ones(model[n-1].shape)
    n+=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    for i in range(3):
        pl.figure()
        pl.imshow(components[i],origin='lower',interpolation='nearest')
        pl.colorbar()
    model = components.sum(0)
    NotPlicely(image,model,sigma)
    
## also print new source params
d = []
l,u = [], []
for key in dic.keys():
    f = dic[key][:,0].reshape((trace.shape[0]*trace.shape[2]))
    lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
    d.append((key,med))
    l.append((key,lo))
    u.append((key,up))

Ddic = dict(d)                    
Ldic = dict(l)
Udic = dict(u)

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|c|cccccc|}\hline'
print r' object & x & y & re & n & pa & q \\\hline'
print 'source 1 & $', '%.2f'%(Ddic['Source 1 x']+lenses[0].pars['x']), '_{-', '%.2f'%Ldic['Source 1 x'],'}^{+','%.2f'%Udic['Source 1 x'], '}$ & $', '%.2f'%(Ddic['Source 1 y']+lenses[0].pars['y']),'_{-', '%.2f'%Ldic['Source 1 y'],'}^{+', '%.2f'%Udic['Source 1 y'], '}$ & $', '%.2f'%Ddic['Source 1 re'],'_{-', '%.2f'%Ldic['Source 1 re'],'}^{+', '%.2f'%Udic['Source 1 re'], '}$ & $', '%.2f'%Ddic['Source 1 n'],'_{-', '%.2f'%Ldic['Source 1 n'],'}^{+', '%.2f'%Udic['Source 1 n'], '}$ & $','%.2f'%Ddic['Source 1 pa'],'_{-', '%.2f'%Ldic['Source 1 pa'],'}^{+', '%.2f'%Udic['Source 1 pa'], '}$ & $','%.2f'%Ddic['Source 1 q'],'_{-', '%.2f'%Ldic['Source 1 q'],'}^{+', '%.2f'%Udic['Source 1 q'], '}$',r'\\'
###
print 'source 2 & $', '%.2f'%(Ddic['Source 2 x']+lenses[0].pars['x']), '_{-', '%.2f'%Ldic['Source 2 x'],'}^{+','%.2f'%Udic['Source 2 x'], '}$ & $', '%.2f'%(Ddic['Source 2 y']+lenses[0].pars['y']),'_{-', '%.2f'%Ldic['Source 2 y'],'}^{+', '%.2f'%Udic['Source 2 y'], '}$ & $', '%.2f'%Ddic['Source 2 re'],'_{-', '%.2f'%Ldic['Source 2 re'],'}^{+', '%.2f'%Udic['Source 2 re'], '}$ & $', '%.2f'%Ddic['Source 2 n'],'_{-', '%.2f'%Ldic['Source 2 n'],'}^{+', '%.2f'%Udic['Source 2 n'], '}$ & $','%.2f'%Ddic['Source 2 pa'],'_{-', '%.2f'%Ldic['Source 2 pa'],'}^{+', '%.2f'%Udic['Source 2 pa'], '}$ & $','%.2f'%Ddic['Source 2 q'],'_{-', '%.2f'%Ldic['Source 2 q'],'}^{+', '%.2f'%Udic['Source 2 q'], '}$',r'\\'
###
print 'galaxy 1 & $', '%.2f'%(Ddic['Galaxy 1 x']), '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%(Ddic['Galaxy 1 y']),'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 1 re'],'_{-', '%.2f'%Ldic['Galaxy 1 re'],'}^{+', '%.2f'%Udic['Galaxy 1 re'], '}$ & $', '%.2f'%Ddic['Galaxy 1 n'],'_{-', '%.2f'%Ldic['Galaxy 1 n'],'}^{+', '%.2f'%Udic['Galaxy 1 n'], '}$ & $','%.2f'%Ddic['Galaxy 1 pa'],'_{-', '%.2f'%Ldic['Galaxy 1 pa'],'}^{+', '%.2f'%Udic['Galaxy 1 pa'], '}$ & $','%.2f'%Ddic['Galaxy 1 q'],'_{-', '%.2f'%Ldic['Galaxy 1 q'],'}^{+', '%.2f'%Udic['Galaxy 1 q'], '}$',r'\\'
###
#print 'galaxy 2 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 2 re'],'_{-', '%.2f'%Ldic['Galaxy 2 re'],'}^{+', '%.2f'%Udic['Galaxy 2 re'], '}$ & $', '%.2f'%Ddic['Galaxy 2 n'],'_{-', '%.2f'%Ldic['Galaxy 2 n'],'}^{+', '%.2f'%Udic['Galaxy 2 n'], '}$ & $','%.2f'%Ddic['Galaxy 2 pa'],'_{-', '%.2f'%Ldic['Galaxy 2 pa'],'}^{+', '%.2f'%Udic['Galaxy 2 pa'], '}$ & $','%.2f'%Ddic['Galaxy 2 q'],'_{-', '%.2f'%Ldic['Galaxy 2 q'],'}^{+', '%.2f'%Udic['Galaxy 2 q'], '}$',r'\\'
###
print 'lens 1 & $', '%.2f'%Ddic['Lens 1 x'], '_{-', '%.2f'%Ldic['Lens 1 x'],'}^{+','%.2f'%Udic['Lens 1 x'], '}$ & $', '%.2f'%Ddic['Lens 1 y'],'_{-', '%.2f'%Ldic['Lens 1 y'],'}^{+', '%.2f'%Udic['Lens 1 y'], '}$ & $', '%.2f'%Ddic['Lens 1 b'],'_{-', '%.2f'%Ldic['Lens 1 b'],'}^{+', '%.2f'%Udic['Lens 1 b'], '}$ & $', '%.2f'%Ddic['Lens 1 eta'],'_{-', '%.2f'%Ldic['Lens 1 eta'],'}^{+', '%.2f'%Udic['Lens 1 eta'], '}$ & $','%.2f'%Ddic['Lens 1 pa'],'_{-', '%.2f'%Ldic['Lens 1 pa'],'}^{+', '%.2f'%Udic['Lens 1 pa'], '}$ & $','%.2f'%Ddic['Lens 1 q'],'_{-', '%.2f'%Ldic['Lens 1 q'],'}^{+', '%.2f'%Udic['Lens 1 q'], '}$',r'\\\hline'
###
#print 'lens blob & $', '%.2f'%Ddic['blob x'], '_{-', '%.2f'%Ldic['blob x'],'}^{+','%.2f'%Udic['blob x'], '}$ & $', '%.2f'%Ddic['blob y'],'_{-', '%.2f'%Ldic['blob y'],'}^{+', '%.2f'%Udic['blob y'], '}$ & $', '%.2f'%Ddic['blob b'],'_{-', '%.2f'%Ldic['blob b'],'}^{+', '%.2f'%Udic['blob b'], '}$ & $', '%.2f'%Ddic['blob eta'],'_{-', '%.2f'%Ldic['blob eta'],'}^{+', '%.2f'%Udic['blob eta'], '}$ & -- & -- ' ,r'\\\hline'
print r'\end{tabular}'
print r'\end{table}'

pl.figure()
pl.plot(lp[:,0])
