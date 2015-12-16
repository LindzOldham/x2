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
    py.writeto('/data/ljo31/Lens/J1125/resid.fits',(image-im)/sigma,clobber=True)
    py.writeto('/data/ljo31/Lens/J1125/model.fits',im,clobber=True)

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

img = py.open('/data/ljo31/Lens/J0837/J0837_Kp_narrow_med.fits')[0].data.copy()[810:1100,790:1105]    #[790:1170,700:1205]
sig = np.ones(img.shape) 
psf = py.open('/data/ljo31/Lens/J0837/PSF_nirc2_Kp_narrow.fits')[0].data.copy()[25:-25,25:-25]
psf = psf/np.sum(psf)



result = np.load('/data/ljo31/Lens/J0837/emcee_Kp0')

lp= result[0]
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]
print lp.shape, trace.shape

imgs = [img]
sigs = [sig]
psfs = [psf]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img.shape,OVRS)
yo,xo = iT.overSample(img.shape,1)
mask = np.ones(img.shape)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(yc,xc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

### first parameters need to be the offsets
#xoffset =  dic['xoffset'][a1,a2,a3]
#yoffset = dic['yoffset'][a1,a2,a3]


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

srcs = []
for name in ['Source 1']:
    p = {}
    if name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
            #p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))


colours = ['F606W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
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
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=11) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=11)
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
    pl.suptitle(str(colours[i]))
    pl.show()
    comps = True
    if comps == True:
        for i in range(len(gals)+len(srcs)):
            pl.figure()
            pl.imshow(components[i],interpolation='nearest',origin='lower',cmap='afmhot')
            pl.colorbar()
    fits.append(fit)



#dx,dy = dic['xoffset'][a1,a2,a3], dic['yoffset'][a1,a2,a3]
x1,y1,re1,n1,pa1,q1 = dic['Source 1 x'][a1,a2,a3], dic['Source 1 y'][a1,a2,a3], dic['Source 1 re'][a1,a2,a3], dic['Source 1 n'][a1,a2,a3], dic['Source 1 pa'][a1,a2,a3], dic['Source 1 q'][a1,a2,a3]
#re5,n5,pa5,q5 = dic['Source 2 re'][a1,a2,a3], dic['Source 2 n'][a1,a2,a3], dic['Source 2 pa'][a1,a2,a3], dic['Source 2 q'][a1,a2,a3]
x2,y2,re2,n2,pa2,q2 = dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3], dic['Galaxy 1 re'][a1,a2,a3], dic['Galaxy 1 n'][a1,a2,a3], dic['Galaxy 1 pa'][a1,a2,a3], dic['Galaxy 1 q'][a1,a2,a3]
#re3,n3,pa3,q3,x3,y3 = dic['Galaxy 2 re'][a1,a2,a3], dic['Galaxy 2 n'][a1,a2,a3], dic['Galaxy 2 pa'][a1,a2,a3], dic['Galaxy 2 q'][a1,a2,a3], dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3]
x4,y4,b,eta,pa4,q4 = dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3], dic['Lens 1 b'][a1,a2,a3], dic['Lens 1 eta'][a1,a2,a3], dic['Lens 1 pa'][a1,a2,a3], dic['Lens 1 q'][a1,a2,a3]
shear,shearpa = dic['extShear'][a1,a2,a3], dic['extShear PA'][a1,a2,a3]

x1,y1 = x1+x4, y1+y4

print 'source 1 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n1, '&', '%.2f'%re1, '&', '%.2f'%q1, '&','%.2f'%pa1,  r'\\'
#print 'source 2 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n5, '&', '%.2f'%re5, '&', '%.2f'%q5, '&','%.2f'%pa5,  r'\\'
print 'galaxy 1 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n2, '&', '%.2f'%re2, '&', '%.2f'%q2, '&','%.2f'%pa2,  r'\\'
#print 'galaxy 2 ', '&', '%.2f'%x3, '&',  '%.2f'%y3, '&', '%.2f'%n3, '&', '%.2f'%re3, '&', '%.2f'%q3, '&','%.2f'%pa3,  r'\\'
print 'lens 1 ', '&', '%.2f'%x4, '&',  '%.2f'%y4, '&', '%.2f'%eta, '&', '%.2f'%b, '&', '%.2f'%q4, '&','%.2f'%pa4,  r'\\\hline'
print 'shear = ', '%.4f'%shear, 'shear pa = ', '%.2f'%shearpa

pl.figure()
pl.plot(lp[:,0])
