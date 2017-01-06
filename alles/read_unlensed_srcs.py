import numpy as np, pylab as pl, pyfits as py
from imageSim import SBObjects, convolve
from astropy.io import fits
from astropy.wcs import WCS
import indexTricks as iT
import pymc
from scipy import optimize,ndimage
import myEmcee_blobs as myEmcee
import cPickle
from linslens import EELsModels as L, Plotter

def NotPlicely(image,im,sigma,name):
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
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-2.5,vmax=2.5,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.suptitle(name)

### use our best models for each source
results = ['J0837_211','J0901_211','J0913_212','/dump/J1125_212_nonconcentric','J1144_212_allparams','J1218_211','J1323_212','J1347_112','J1446_212','J1605_212','J1606_112','J2228_212']
names = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']


for i in range(4,5):
    name=names[i]
    print name
    result = np.load('/data/ljo31/Lens/LensModels/'+results[i])
    model = L.EELs(result,name)
    model.Initialise()
    V,I = model.unlensedeval()
    whtV,whtI = model.AddWeightImages()
    #PSFs = model.PSFs
    psf1,psf2 = model.psf1,model.psf2
    psfs=[psf1,psf2]
    ## add noise to image abd create noisemaps
    sigmaV,sigmaI = np.mean(model.sig1[:10,:10]),np.mean(model.sig2[:10,:10])
    V, I = V + np.random.randn(V.shape[0],V.shape[1])*sigmaV, I + np.random.randn(I.shape[0],I.shape[1])*sigmaI
    smoothV, smoothI = ndimage.gaussian_filter(V,0.7),ndimage.gaussian_filter(I,0.7) 
    sig1 = np.where((smoothV>0.7*sigmaV)&(V>0),V/whtV+sigmaV**2., sigmaV**2)**0.5
    sig2 = np.where((smoothI>0.7*sigmaI)&(I>0),I/whtI+sigmaI**2., sigmaI**2)**0.5
    sig1[np.isnan(sig1)] = 1
    sig2[np.isnan(sig2)] = 1
    sig1[~np.isfinite(sig1)] = 1
    sig2[~np.isfinite(sig2)] = 1
    imgs = [V,I]
    sigs = [sig1,sig2]
    yo,xo = iT.overSample(V.shape,1)
    x0,y0 = 0.5*np.amax(xo),0.5*np.amax(yo)
    PSFs = []
    for i in range(len(imgs)):
        psf = psfs[i]
        image = imgs[i]
        psf /= psf.sum()
        psf = convolve.convolve(image,psf)[1]
        PSFs.append(psf)
    kres = np.load('/data/ljo31/Lens/Analysis/fit_unlensed_src_5_'+str(name))
    lp,trace,dic,_ = kres
    a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    a2=0
    x,y,pa,q,re,n,dx,dy = trace[a1,a2,a3]
    gal = SBObjects.Sersic('Galaxy',{'x':x,'y':y,'pa':pa,'q':q,'re':re,'n':n})
    gal.setPars()
    for i in range(2):
        if i == 0:
            xp,yp = xo,yo
        else:
            xp,yp=xo+dx,yo+dy
        image,sigma,psf = imgs[i],sigs[i],PSFs[i]
        imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
        model = np.zeros((1,imin.size))
        tmp = xp*0.
        tmp = gal.pixeval(xin,yin,csub=23).reshape(xp.shape)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[0] = tmp.ravel()
        rhs = (imin/sigin) 
        op = (model/sigin).T
        fit, chi = optimize.nnls(op,rhs)
        components = (model.T*fit).T.reshape((1,image.shape[0],image.shape[1]))
        model = components.sum(0)
        Plotter.SotPleparately(image,model,sigma,name,vmin=-2.5,vmax=2.5)
        #NotPlicely(image,model,sigma,name)
        pl.show()
'''
 outFile = '/data/ljo31b/MACSJ0717/models/model_'+str(name)+'_'+str(i)+'.fits'
 f = open(outFile,'wb')
 cPickle.dump(model,f,2)
    f.close()
    outFile = '/data/ljo31b/MACSJ0717/models/fit_'+str(name)+'_'+str(i)
    f = open(outFile,'wb')
    cPickle.dump(fit,f,2)
    f.close()
'''
