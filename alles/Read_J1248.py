import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
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
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.5) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=np.amax(image)*0.5) #,vmin=vmin,vmax=vmax)
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



img1 = py.open('/data/ljo31/Lens/J1248/F555W_sci_cutout.fits')[0].data.copy()[10:-10,20:-25]
sig1 = py.open('/data/ljo31/Lens/J1248/F555W_noisemap.fits')[0].data.copy()[10:-10,20:-25]
psf1 = py.open('/data/ljo31/Lens/J1248/F555W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1248/F814W_sci_cutout.fits')[0].data.copy()[10:-10,20:-25]
#sig2 = np.ones(img1.shape)
#sig2 = py.open('/data/ljo31/Lens/J1248/F814W_noisemap.fits')[0].data.copy()[10:-10,20:-25]
psf2 = py.open('/data/ljo31/Lens/J1248/F814W_psf1.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)
#psf2 = py.open('/data/ljo31/Lens/J1248/F555W_psf1.fits')[0].data.copy()
#psf2=psf2/np.sum(psf2)
result = np.load('/data/ljo31/Lens/J1248/new1')
lp,trace,dic,_ = result
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 1
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xo,xc=xo+10,xc+10
mask = np.zeros(img1.shape)
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

gal1 = SBModels.Sersic('Galaxy 1',{'x':dic['Galaxy 1 x'][a1,0,a3],'y':dic['Galaxy 1 y'][a1,0,a3],'n':dic['Galaxy 1 n'][a1,0,a3],'re':dic['Galaxy 1 re'][a1,0,a3],'pa':dic['Galaxy 1 pa'][a1,0,a3],'q':dic['Galaxy 1 q'][a1,0,a3]})
gal2 = SBModels.Sersic('Galaxy 2',{'x':dic['Galaxy 2 x'][a1,0,a3],'y':dic['Galaxy 2 y'][a1,0,a3],'n':dic['Galaxy 2 n'][a1,0,a3],'re':dic['Galaxy 2 re'][a1,0,a3],'pa':dic['Galaxy 2 pa'][a1,0,a3],'q':dic['Galaxy 2 q'][a1,0,a3]})
gals = [gal1,gal2]

src1 = SBModels.Sersic('Source 1',{'x':dic['Source 1 x'][a1,0,a3],'y':dic['Source 1 y'][a1,0,a3],'n':dic['Source 1 n'][a1,0,a3],'re':dic['Source 1 re'][a1,0,a3],'pa':dic['Source 1 pa'][a1,0,a3],'q':dic['Source 1 q'][a1,0,a3]})
srcs = [src1]

lens1 = MassModels.PowerLaw('Lens 1',{'x':dic['Lens 1 x'][a1,0,a3],'y':dic['Lens 1 y'][a1,0,a3],'b':dic['Lens 1 b'][a1,0,a3],'eta':dic['Lens 1 eta'][a1,0,a3],'q':dic['Lens 1 q'][a1,0,a3],'pa':dic['Lens 1 pa'][a1,0,a3]})
shear = MassModels.ExtShear('shear',{'x':dic['Lens 1 x'][a1,0,a3],'y':dic['Lens 1 y'][a1,0,a3],'b':0.06,'pa':-50.})#dic['extShear'][a1,0,a3],'pa':dic['extShear PA'][a1,0,a3]})
lenses = [lens1,shear]

xoffset, yoffset = dic['xoffset'][a1,0,a3],dic['yoffset'][a1,0,a3]

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
    xop,yop = xo+dx,yo+dy
    image,sigma,psf = imgs[i],sigs[i],PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=21)
        tmp = iT.resamp(tmp,OVRS,True) 
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
    comps = False
    if comps == True:
        CotSomponents(components,colours[i])
    fits.append(fit)



#dx,dy = dic['xoffset'][a1,a2,a3], dic['yoffset'][a1,a2,a3]
x1,y1,re1,n1,pa1,q1 = dic['Source 1 x'][a1,a2,a3], dic['Source 1 y'][a1,a2,a3], dic['Source 1 re'][a1,a2,a3], dic['Source 1 n'][a1,a2,a3], dic['Source 1 pa'][a1,a2,a3], dic['Source 1 q'][a1,a2,a3]
#re5,n5,pa5,q5,x5,y5 = dic['Source 2 re'][a1,a2,a3], dic['Source 2 n'][a1,a2,a3], dic['Source 2 pa'][a1,a2,a3], dic['Source 2 q'][a1,a2,a3],dic['Source 2 x'][a1,a2,a3], dic['Source 2 y'][a1,a2,a3],
x2,y2,re2,n2,pa2,q2 = dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3], dic['Galaxy 1 re'][a1,a2,a3], dic['Galaxy 1 n'][a1,a2,a3], dic['Galaxy 1 pa'][a1,a2,a3], dic['Galaxy 1 q'][a1,a2,a3]
re3,n3,pa3,q3,x3,y3 = dic['Galaxy 2 re'][a1,a2,a3], dic['Galaxy 2 n'][a1,a2,a3], dic['Galaxy 2 pa'][a1,a2,a3], dic['Galaxy 2 q'][a1,a2,a3], dic['Galaxy 2 x'][a1,a2,a3], dic['Galaxy 2 y'][a1,a2,a3]
x4,y4,b,eta,pa4,q4 = dic['Lens 1 x'][a1,a2,a3], dic['Lens 1 y'][a1,a2,a3], dic['Lens 1 b'][a1,a2,a3], dic['Lens 1 eta'][a1,a2,a3], dic['Lens 1 pa'][a1,a2,a3], dic['Lens 1 q'][a1,a2,a3]
#shear,shearpa = dic['extShear'][a1,a2,a3], dic['extShear PA'][a1,a2,a3]
#x4a,y4a,ba,etaa,pa4a,q4a = dic['Lens 2 x'][a1,a2,a3], dic['Lens 2 y'][a1,a2,a3], dic['Lens 2 b'][a1,a2,a3], dic['Lens 2 eta'][a1,a2,a3], dic['Lens 2 pa'][a1,a2,a3], dic['Lens 2 q'][a1,a2,a3]

#x5,y5 = x5+x4, y5+y4

print 'source 1 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n1, '&', '%.2f'%re1, '&', '%.2f'%q1, '&','%.2f'%pa1,  r'\\'
#print 'source 2 ', '&', '%.2f'%x5, '&',  '%.2f'%y5, '&', '%.2f'%n5, '&', '%.2f'%re5, '&', '%.2f'%q5, '&','%.2f'%pa5,  r'\\'
print 'galaxy 1 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n2, '&', '%.2f'%re2, '&', '%.2f'%q2, '&','%.2f'%pa2,  r'\\'
print 'galaxy 2 ', '&', '%.2f'%x3, '&',  '%.2f'%y3, '&', '%.2f'%n3, '&', '%.2f'%re3, '&', '%.2f'%q3, '&','%.2f'%pa3,  r'\\'
print 'lens 1 ', '&', '%.2f'%x4, '&',  '%.2f'%y4, '&', '%.2f'%eta, '&', '%.2f'%b, '&', '%.2f'%q4, '&','%.2f'%pa4,  r'\\'
#print 'lens 2 ', '&', '%.2f'%x4a, '&',  '%.2f'%y4a, '&', '%.2f'%etaa, '&', '%.2f'%ba, '&', '%.2f'%q4a, '&','%.2f'%pa4a,  r'\\\hline'
#print 'shear = ', '%.4f'%shear, 'shear pa = ', '%.2f'%shearpa

pl.figure()
pl.plot(lp[:,0])
