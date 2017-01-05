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


image = py.open('/data/ljo31/Lens/J0837/J0837_Kp_narrow_med.fits')[0].data.copy()[810:1100,790:1105]    #[790:1170,700:1205]
sigma = np.ones(image.shape) 

name='J0837'
dir = '/data/ljo31/Lens/LensModels/twoband/'
result = np.load(dir+name+'_211')
kresult = np.load(dir+name+'_Kp_211')

lp,trace,dic,_ = result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

klp,ktrace,kdic,_=kresult
ka2=0
ka1,ka3 = numpy.unravel_index(klp[:,0].argmax(),klp[:,0].shape)

OVRS = 5
yc,xc = iT.overSample(image.shape,OVRS)
yo,xo = iT.overSample(image.shape,1)
xc,xo,yc,yo=xc*0.2,xo*0.2,yc*0.2,yo*0.2
xc,xo = xc+16 , xo+16 # check offsets
yc,yo = yc+23 , yo+23 # 
mask = np.zeros(image.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

imgs = [image]
sigs = [sigma]
img=image.copy()
xpsf,ypsf = iT.coords((101,101))-50

xoffset,yoffset,sig1,q1,pa1,amp1,sig2,q2,pa2,amp2,sig3,q3,pa3,amp3 = ktrace[ka1,ka2,ka3,:14]

# working on a 0.05 arcsec per pixel scale. 50 mas. psf=100mas for LGS.
psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':2.,'q':1,'pa':0,'amp':1})
psf1 = psfObj1.pixeval(xpsf,ypsf)
psf = psf1/psf1.sum()

psf = convolve.convolve(img,psf)[1]
PSFs=[psf]


gals = []
for name in ['Galaxy 1', 'Galaxy 2']:
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
for name in ['Source 2', 'Source 1']:
    p = {}
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           p[key] = kdic[name+' '+key][ka1,ka2,ka3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = kdic[name+' '+key][ka1,ka2,ka3]#+lenses[0].pars[key]
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           p[key] = kdic[name+' '+key][ka1,ka2,ka3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = kdic[name+' '+key][ka1,ka2,ka3]+lenses[0].pars[key]
            #p[key] = srcs[0].pars[key]
    srcs.append(SBBModels.Sersic(name,p))


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
        tmp[mask2] = gal.pixeval(xin,yin,0.2,csub=31) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp[mask2] = src.pixeval(x0,y0,0.2,csub=31)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        if src.name== 'Source 2':
            print 'dust lane'
            model[n] *= -1
        n +=1
    model[n] = -1*np.ones(model[n-1].size)
    n+=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    

ZPs = np.load('/data/ljo31/Lens/LensParams/Keck_zeropoints.npy')[()]


src = components[3]

xap1 = [88,100]
yap1 = [105,120]
yap2 = [138,155]
xap2 = [90,110]

area = (xap1[1]-xap1[0]) * (yap1[1]-yap1[0]) + (xap2[1]-xap2[0]) * (yap2[1]-yap2[0]) # in pixels, 0.05 arcsec per pixel
area *= 0.01 # in arcsec squared
print area

flux = np.sum(src[yap1[0]:yap1[1],xap1[0]:xap1[1]])
flux += np.sum(src[yap2[0]:yap2[1],xap2[0]:xap2[1]])

src[yap1[0]:yap1[1],xap1[0]:xap1[1]] = 0
src[yap2[0]:yap2[1],xap2[0]:xap2[1]] = 0
mag = -2.5*np.log10(flux) + ZPs['J0837'] #+ 2.5*np.log10(area)
print mag

pl.imshow(src,origin='lower')

print flux
