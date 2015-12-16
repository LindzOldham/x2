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
import SBBModels, SBBProfiles


# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.3) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.3) #,vmin=vmin,vmax=vmax)
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
    py.writeto('/data/ljo31/Lens/J0837/resid.fits',(image-im),clobber=True)
    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')

def SotPleparately(image,im,sigma,col):
    ext = [0,image.shape[0],0,image.shape[1]]
    pl.figure()
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data - '+str(col))
    pl.figure()
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model - '+str(col))
    pl.figure()
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot')
    pl.title('signal-to-noise residuals - '+str(col))
    pl.colorbar()

def CotSomponents(components,col):
    pl.figure()
    pl.subplot(221)
    pl.imshow(components[0],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto',vmax=np.amax(components[0])*0.5)
    pl.colorbar()
    pl.title('galaxy 1 ')
    pl.subplot(222)
    pl.imshow(components[1],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto',vmax=np.amax(components[1])*0.5)
    pl.colorbar()
    pl.title('galaxy 2 ')
    pl.subplot(223)
    pl.imshow(components[2],interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto',vmax=np.amax(components[2])*0.5)
    pl.colorbar()
    pl.title('source 1 ')
    pl.suptitle(col)
    pl.subplot(224)
    pl.imshow(components.sum(0),interpolation='nearest',origin='lower',cmap='afmhot',aspect='auto',vmax=np.amax(components[2])*0.5)
    pl.colorbar()
    pl.title('model ')
    pl.suptitle(col)


img1 = py.open('/data/ljo31/Lens/J1125/F606W_sci_cutout.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1125/F606W_noisemap_edited.fits')[0].data.copy()
psf1 = py.open('/data/ljo31/Lens/J1125/F606W_psf3_filledin.fits')[0].data.copy()
psf1 = psf1[5:-7,5:-6]
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1125/F814W_sci_cutout.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1125/F814W_noisemap_edited.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1125/F814W_psf3_filledin.fits')[0].data.copy()
psf2 = psf2[5:-8,5:-6]
psf2 = psf2/np.sum(psf2)


result = np.load('/data/ljo31/Lens/J1125/emcee_1src010')
#result = np.load('/data/ljo31/Lens/J1125/emcee_1src1')
result = np.load('/data/ljo31/Lens/J1125/emcee_1src2')

lp= result[0]
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]
print lp.shape, trace.shape

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
mask = py.open('/data/ljo31/Lens/J1125/mask814.fits')[0].data.copy()
#mask = np.ones(img1.shape)
tck = RectBivariateSpline(xo[0],yo[:,0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2.T
mask2 = mask2==1
mask = mask==1

## also try masking this bad region:
maskX = py.open('/data/ljo31/Lens/J1125/mask_badpix.fits')[0].data.copy() + py.open('/data/ljo31/Lens/J1125/mask_badpix2.fits')[0].data.copy() # masking two areas now
tck = RectBivariateSpline(xo[0],yo[:,0],maskX)
mask2X = tck.ev(xc,yc)
mask2X[mask2X<0.5] = 0
mask2X[mask2X>0.5] = 1
mask2X = mask2X.T

mask2 = ((mask2==1) & (mask2X==0))
mask = ((mask==1) & (maskX==0))
print img1[mask].shape # should be 6678


for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

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
            p[key] = gals[0].pars[key]
    gals.append(SBBModels.Sersic(name,p))

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

srcs = []
for name in ['Source 1']:
    p = {}
    if name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
    srcs.append(SBBModels.Sersic(name,p))


colours = ['F606W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
    if i == 0:
        dx,dy = 0,0
    else:
        dx = dic['xoffset'][a1,a2,a3]
        dy = dic['yoffset'][a1,a2,a3]
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    #cc=dic['boxiness'][a1,a2,a3]
    #print cc
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.boxypixeval(xp,yp,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp = src.boxypixeval(x0,y0,1./OVRS,csub=11)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n] = np.ones(model[n].shape)
    n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.show()
    comps = True
    if comps == True:
        CotSomponents(components,colours[i])
    fits.append(fit)

print fits

#dx,dy = dic['xoffset'][a1,a2,a3], dic['yoffset'][a1,a2,a3]
x1,y1,re1,n1,pa1,q1 = dic['Source 1 x'][a1,a2,a3], dic['Source 1 y'][a1,a2,a3], dic['Source 1 re'][a1,a2,a3], dic['Source 1 n'][a1,a2,a3], dic['Source 1 pa'][a1,a2,a3], dic['Source 1 q'][a1,a2,a3]
x2,y2,re2,n2,pa2,q2 = dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3], dic['Galaxy 1 re'][a1,a2,a3], dic['Galaxy 1 n'][a1,a2,a3], dic['Galaxy 1 pa'][a1,a2,a3], dic['Galaxy 1 q'][a1,a2,a3]
#re3,n3,pa3,q3,x3,y3 = dic['Galaxy 2 re'][a1,a2,a3], dic['Galaxy 2 n'][a1,a2,a3], dic['Galaxy 2 pa'][a1,a2,a3], dic['Galaxy 2 q'][a1,a2,a3], dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3]
x4,y4,b,eta,pa4,q4 = dic['Lens 1 x'][a1,a2,a3], dic['Lens 1 y'][a1,a2,a3], dic['Lens 1 b'][a1,a2,a3], dic['Lens 1 eta'][a1,a2,a3], dic['Lens 1 pa'][a1,a2,a3], dic['Lens 1 q'][a1,a2,a3]
shear,shearpa = dic['extShear'][a1,a2,a3], dic['extShear PA'][a1,a2,a3]
#x4a,y4a,ba,etaa,pa4a,q4a = dic['Lens 2 x'][a1,a2,a3], dic['Lens 2 y'][a1,a2,a3], dic['Lens 2 b'][a1,a2,a3], dic['Lens 2 eta'][a1,a2,a3], dic['Lens 2 pa'][a1,a2,a3], dic['Lens 2 q'][a1,a2,a3]
#x6,y6,re6,n6,pa6,q6 = dic['Source 2 x'][a1,a2,a3], dic['Source 2 y'][a1,a2,a3], dic['Source 2 re'][a1,a2,a3], dic['Source 2 n'][a1,a2,a3], dic['Source 2 pa'][a1,a2,a3], dic['Source 2 q'][a1,a2,a3]

x1,y1 = x1+x4, y1+y4
#x6,y6 = x6+x4, y6+y4

print 'source 1 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n1, '&', '%.2f'%re1, '&', '%.2f'%q1, '&','%.2f'%pa1,  r'\\'
#print 'source 2 ', '&', '%.2f'%x6, '&',  '%.2f'%y6, '&', '%.2f'%n6, '&', '%.2f'%re6, '&', '%.2f'%q6, '&','%.2f'%pa6,  r'\\'
print 'galaxy 1 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n2, '&', '%.2f'%re2, '&', '%.2f'%q2, '&','%.2f'%pa2,  r'\\'
#print 'galaxy 2 ', '&', '%.2f'%x3, '&',  '%.2f'%y3, '&', '%.2f'%n3, '&', '%.2f'%re3, '&', '%.2f'%q3, '&','%.2f'%pa3,  r'\\'
print 'lens 1 ', '&', '%.2f'%x4, '&',  '%.2f'%y4, '&', '%.2f'%eta, '&', '%.2f'%b, '&', '%.2f'%q4, '&','%.2f'%pa4,  r'\\\hline'
#print 'lens 2 ', '&', '%.2f'%x4a, '&',  '%.2f'%y4a, '&', '%.2f'%etaa, '&', '%.2f'%ba, '&', '%.2f'%q4a, '&','%.2f'%pa4a,  r'\\\hline'
print 'shear = ', '%.4f'%shear, 'shear pa = ', '%.2f'%shearpa

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
print 'source 1 & $', '%.2f'%(Ddic['Source 1 x']+Ddic['Lens 1 x']), '_{-', '%.2f'%Ldic['Source 1 x'],'}^{+','%.2f'%Udic['Source 1 x'], '}$ & $', '%.2f'%(Ddic['Source 1 y']+Ddic['Lens 1 y']),'_{-', '%.2f'%Ldic['Source 1 y'],'}^{+', '%.2f'%Udic['Source 1 y'], '}$ & $', '%.2f'%Ddic['Source 1 re'],'_{-', '%.2f'%Ldic['Source 1 re'],'}^{+', '%.2f'%Udic['Source 1 re'], '}$ & $', '%.2f'%Ddic['Source 1 n'],'_{-', '%.2f'%Ldic['Source 1 n'],'}^{+', '%.2f'%Udic['Source 1 n'], '}$ & $','%.2f'%Ddic['Source 1 pa'],'_{-', '%.2f'%Ldic['Source 1 pa'],'}^{+', '%.2f'%Udic['Source 1 pa'], '}$ & $','%.2f'%Ddic['Source 1 q'],'_{-', '%.2f'%Ldic['Source 1 q'],'}^{+', '%.2f'%Udic['Source 1 q'], '}$',r'\\'
###
print 'galaxy 1 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 1 re'],'_{-', '%.2f'%Ldic['Galaxy 1 re'],'}^{+', '%.2f'%Udic['Galaxy 1 re'], '}$ & $', '%.2f'%Ddic['Galaxy 1 n'],'_{-', '%.2f'%Ldic['Galaxy 1 n'],'}^{+', '%.2f'%Udic['Galaxy 1 n'], '}$ & $','%.2f'%Ddic['Galaxy 1 pa'],'_{-', '%.2f'%Ldic['Galaxy 1 pa'],'}^{+', '%.2f'%Udic['Galaxy 1 pa'], '}$ & $','%.2f'%Ddic['Galaxy 1 q'],'_{-', '%.2f'%Ldic['Galaxy 1 q'],'}^{+', '%.2f'%Udic['Galaxy 1 q'], '}$',r'\\'
###
#print 'galaxy 2 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 2 re'],'_{-', '%.2f'%Ldic['Galaxy 2 re'],'}^{+', '%.2f'%Udic['Galaxy 2 re'], '}$ & $', '%.2f'%Ddic['Galaxy 2 n'],'_{-', '%.2f'%Ldic['Galaxy 2 n'],'}^{+', '%.2f'%Udic['Galaxy 2 n'], '}$ & $','%.2f'%Ddic['Galaxy 2 pa'],'_{-', '%.2f'%Ldic['Galaxy 2 pa'],'}^{+', '%.2f'%Udic['Galaxy 2 pa'], '}$ & $','%.2f'%Ddic['Galaxy 2 q'],'_{-', '%.2f'%Ldic['Galaxy 2 q'],'}^{+', '%.2f'%Udic['Galaxy 2 q'], '}$',r'\\'
###
print 'lens 1 & $', '%.2f'%Ddic['Lens 1 x'], '_{-', '%.2f'%Ldic['Lens 1 x'],'}^{+','%.2f'%Udic['Lens 1 x'], '}$ & $', '%.2f'%Ddic['Lens 1 y'],'_{-', '%.2f'%Ldic['Lens 1 y'],'}^{+', '%.2f'%Udic['Lens 1 y'], '}$ & $', '%.2f'%Ddic['Lens 1 b'],'_{-', '%.2f'%Ldic['Lens 1 b'],'}^{+', '%.2f'%Udic['Lens 1 b'], '}$ & $', '%.2f'%Ddic['Lens 1 eta'],'_{-', '%.2f'%Ldic['Lens 1 eta'],'}^{+', '%.2f'%Udic['Lens 1 eta'], '}$ & $','%.2f'%Ddic['Lens 1 pa'],'_{-', '%.2f'%Ldic['Lens 1 pa'],'}^{+', '%.2f'%Udic['Lens 1 pa'], '}$ & $','%.2f'%Ddic['Lens 1 q'],'_{-', '%.2f'%Ldic['Lens 1 q'],'}^{+', '%.2f'%Udic['Lens 1 q'], '}$',r'\\\hline'
###
print r'\end{tabular}'
print r'\caption{', 'shear = $', '%.2f'%Ddic['extShear'], '_{-', '%.2f'%Ldic['extShear'],'}^{+','%.2f'%Udic['extShear'], '}$ , shear pa = $',  '%.2f'%Ddic['extShear PA'], '_{-', '%.2f'%Ldic['extShear PA'],'}^{+','%.2f'%Udic['extShear PA'], '}$}'
print r'\end{table}'
