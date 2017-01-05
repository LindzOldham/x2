import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles

# plot things
def NotPlicely(image,im,sigma):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.99) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.99) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-50,vmax=50,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()


img1 = py.open('/data/ljo31b/lenses/chip5/imgg.fits')[0].data[450:-450,420:-480]
bg=np.median(img1[-10:,-10:])
img1-=bg
sig1 = py.open('/data/ljo31b/lenses/chip5/noisemap_g_big.fits')[0].data
psf1 = py.open('/data/ljo31b/lenses/g_psf3.fits')[0].data 

img2 = py.open('/data/ljo31b/lenses/chip5/imgr.fits')[0].data[450:-450,420:-480]
bg=np.median(img2[-10:,-10:])
img2-=bg
sig2 = py.open('/data/ljo31b/lenses/chip5/noisemap_r_big.fits')[0].data
psf2 = py.open('/data/ljo31b/lenses/r_psf3.fits')[0].data

img3 = py.open('/data/ljo31b/lenses/chip5/imgi.fits')[0].data[450:-450,420:-480]
bg=np.median(img3[-10:,-10:])
img3-=bg
sig3 = py.open('/data/ljo31b/lenses/chip5/noisemap_i_big.fits')[0].data
psf3 = py.open('/data/ljo31b/lenses/i_psf3.fits')[0].data


imgs = [img1,img2,img3]
sigs = [sig1,sig2,sig3]
psfs = [psf1,psf2,psf3]
PSFs = []
for i in range(len(psfs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

result = np.load('/data/ljo31b/lenses/model_gri_gri_B_7')
lp,trace,dic,_= result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xo,xc,yo,yc = xo-10,xc-10,yo-10,yc-10

maskg = py.open('/data/ljo31b/lenses/chip5/newmask.fits')[0].data
maski = py.open('/data/ljo31b/lenses/chip5/mask_i.fits')[0].data
maskr = py.open('/data/ljo31b/lenses/chip5/newmasknew.fits')[0].data
mask = np.where((maskg==1)|(maski==1)|(maskr==1),1,0)
tck = RectBivariateSpline(yo[:,0],xo[0],mask)
mask2 = tck.ev(xc,yc)
mask2[mask2<0.5] = 0
mask2[mask2>0.5] = 1
mask2 = mask2==0
mask = mask==0

pars = []
cov = []

# offsets
pars.append(pymc.Uniform('gr xoffset',-5.,5.,value=dic['gr xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('gr yoffset',-5.,5.,value=dic['gr yoffset'][a1,a2,a3]))
cov += [0.1,0.1]
pars.append(pymc.Uniform('gi xoffset',-10.,10.,value=dic['gi xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('gi yoffset',-10.,10.,value=dic['gi yoffset'][a1,a2,a3]))
cov += [0.1,0.1]

name = 'Galaxy 1 '
gal1 = SBModels.Sersic('Galaxy 1',{'x':dic[name+'x'][a1,a2,a3],'y':dic[name+'y'][a1,a2,a3],'re':dic[name+'re'][a1,a2,a3],'n':dic[name+'n'][a1,a2,a3],'q':dic[name+'q'][a1,a2,a3],'pa':dic[name+'pa'][a1,a2,a3]})
name = 'Galaxy 2 '
gal2 = SBModels.Sersic('Galaxy 2',{'x':dic['Galaxy 1 x'][a1,a2,a3],'y':dic['Galaxy 1 y'][a1,a2,a3],'re':dic[name+'re'][a1,a2,a3],'n':dic[name+'n'][a1,a2,a3],'q':dic[name+'q'][a1,a2,a3],'pa':dic[name+'pa'][a1,a2,a3]})
gals = [gal1,gal2]

name = 'Lens 1 '
lens1 = MassModels.PowerLaw('Lens 1',{'x':dic[name+'x'][a1,a2,a3],'y':dic[name+'y'][a1,a2,a3],'b':dic[name+'b'][a1,a2,a3],'eta':dic[name+'eta'][a1,a2,a3],'q':dic[name+'q'][a1,a2,a3],'pa':dic[name+'pa'][a1,a2,a3]})
lens2 = MassModels.ExtShear('shear',{'x':dic[name+'x'][a1,a2,a3],'y':dic[name+'y'][a1,a2,a3],'b':dic['extShear'][a1,a2,a3],'pa':dic['extShear PA'][a1,a2,a3]})
lenses = [lens1,lens2]

name = 'Source 1 '
src1 = SBModels.Sersic('Source 1',{'x':dic[name+'x'][a1,a2,a3],'y':dic[name+'y'][a1,a2,a3],'re':dic[name+'re'][a1,a2,a3],'n':dic[name+'n'][a1,a2,a3],'q':dic[name+'q'][a1,a2,a3],'pa':dic[name+'pa'][a1,a2,a3]})
name = 'Source 2 '
src2 = SBModels.Sersic('Source 2',{'x':dic[name+'x'][a1,a2,a3],'y':dic[name+'y'][a1,a2,a3],'re':dic[name+'re'][a1,a2,a3],'n':dic[name+'n'][a1,a2,a3],'q':dic[name+'q'][a1,a2,a3],'pa':dic[name+'pa'][a1,a2,a3]})
srcs = [src1,src2]

colours = ['g','r','i']
models = []
fits = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    elif i == 1:
        dx = pars[0].value 
        dy = pars[1].value 
    elif i == 2:
        dx = pars[2].value 
        dy = pars[3].value 
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
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=21) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp = src.pixeval(x0,y0,1./OVRS,csub=21)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n]=np.ones(model[n].shape)
    n+=1
    rhs = image[mask]/sigma[mask]
    mmodel = model.reshape((n,image.shape[0],image.shape[1]))
    mmmodel = np.empty(((len(gals) + len(srcs)+1),image[mask].size))
    for m in range(mmodel.shape[0]):
        mmmodel[m] = mmodel[m][mask]
    op = (mmmodel/sigma[mask]).T
    rhs = image[mask]/sigma[mask]
    #pl.imshow(mmodel[0])
    #pl.figure()
    #pl.imshow(mmodel[1])
    #pl.figure()
    #pl.imshow(mmodel[2])
    #pl.figure()
    #pl.imshow(mmodel[3])
    fit, chi = optimize.nnls(op,rhs)
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma)
    pl.suptitle(str(colours[i]))
    pl.show()
       
pl.figure()
pl.plot(lp[:,0])
pl.show()

from tools.simple import climshow
for i in range(len(components)):
    pl.figure()
    climshow(components[i])
    pl.colorbar()

pl.show()
py.writeto('/data/ljo31b/lenses/model_B_gri_gri_2_ctd.fits',image-model,clobber=True)

'''xc,yc=xc[25:-25,25:-25], yc[25:-25,25:-25]
src = fit[-3]*srcs[0].pixeval(xc,yc) + fit[-2]*srcs[1].pixeval(xc,yc)
src0 = fit[-3]*srcs[0].pixeval(xc,yc)
src1 = fit[-2]*srcs[1].pixeval(xc,yc)

pl.figure()
pl.subplot(131)
climshow(src)
pl.colorbar()
pl.subplot(132)
climshow(src0)
pl.colorbar()
pl.subplot(133)
climshow(src1)
pl.colorbar()
pl.suptitle('source plane')
'''
