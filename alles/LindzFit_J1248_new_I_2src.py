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

JJ='new1_Iband_3_2src'
print JJ

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


img1 = py.open('/data/ljo31/Lens/J1248/F555W_sci_cutout.fits')[0].data.copy()[10:-10,20:-25]
sig1 = py.open('/data/ljo31/Lens/J1248/F555W_noisemap.fits')[0].data.copy()[10:-10,20:-25]
psf1 = py.open('/data/ljo31/Lens/J1248/F555W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1248/F814W_sci_cutout.fits')[0].data.copy()[10:-10,20:-25]
sig2 = py.open('/data/ljo31/Lens/J1248/F814W_noisemap.fits')[0].data.copy()[10:-10,20:-25]
psf2 = py.open('/data/ljo31/Lens/J1248/F814W_psf1.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)

result = np.load('/data/ljo31/Lens/J1248/pixsrc_2_ctd_new_Iband')
lp,trace,dic,_ = result
a2=0.
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

gresult = np.load('/data/ljo31/Lens/LensModels/J1248_211_V')
glp,gtrace,gdic,_ = gresult
ga2=0.
ga1,ga3 = numpy.unravel_index(glp[:,0].argmax(),glp[:,0].shape)

imgs = [img2]
sigs = [sig2]
psfs = [psf2]

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

vx,vy = gdic['Galaxy 1 x'][ga1,0,ga3], gdic['Galaxy 1 y'][ga1,0,ga3]
GX1 = vx#pymc.Uniform('Galaxy 1 x',vx-5,vx+5,vx)
GY1 = vy#pymc.Uniform('Galaxy 1 y',vy-5,vy+5,vy)
GN1 = gdic['Galaxy 1 n'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 n',0.5,8.,gdic['Galaxy 1 n'][ga1,0,ga3])
GRE1 = gdic['Galaxy 1 re'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 re',0.5,100.,gdic['Galaxy 1 re'][ga1,0,ga3])
GPA1 = gdic['Galaxy 1 pa'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 pa',0.,180.,gdic['Galaxy 1 pa'][ga1,0,ga3])
GQ1 = gdic['Galaxy 1 q'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 q',0.1,1.,gdic['Galaxy 1 q'][ga1,0,ga3])

vx,vy = gdic['Galaxy 2 x'][ga1,0,ga3], gdic['Galaxy 2 y'][ga1,0,ga3]
GX2 = vx#pymc.Uniform('Galaxy 1 x',vx-5,vx+5,vx)
GY2 = vy#pymc.Uniform('Galaxy 1 y',vy-5,vy+5,vy)
GN2 = gdic['Galaxy 2 n'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 n',0.5,8.,gdic['Galaxy 1 n'][ga1,0,ga3])
GRE2 = gdic['Galaxy 2 re'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 re',0.5,100.,gdic['Galaxy 1 re'][ga1,0,ga3])
GPA2 = gdic['Galaxy 2 pa'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 pa',0.,180.,gdic['Galaxy 1 pa'][ga1,0,ga3])
GQ2 = gdic['Galaxy 2 q'][ga1,0,ga3]#pymc.Uniform('Galaxy 1 q',0.1,1.,gdic['Galaxy 1 q'][ga1,0,ga3])

gal1 = SBModels.Sersic('Galaxy 1',{'x':GX1,'y':GY1,'n':GN1,'re':GRE1,'pa':GPA1,'q':GQ1})
gal2 = SBModels.Sersic('Galaxy 2',{'x':GX2,'y':GY2,'n':GN2,'re':GRE2,'pa':GPA2,'q':GQ2})
gals = [gal1,gal2]

X = pymc.Uniform('Lens 1 x',58.7,68.7,dic['Lens 1 x'][a1,0,a3])
Y = pymc.Uniform('Lens 1 y',53.1,63.1,dic['Lens 1 y'][a1,0,a3])
B = dic['Lens 1 b'][a1,0,a3]#pymc.Uniform('Lens 1 b',0.5,100.,value=9.0)#dic['Lens 1 b'][a1,0,a3])
Q = dic['Lens 1 q'][a1,0,a3]#pymc.Uniform('Lens 1 q',0.1,1.0,value=0.44)#dic['Lens 1 q'][a1,0,a3])
ETA = dic['Lens 1 eta'][a1,0,a3]#pymc.Uniform('Lens 1 eta',0.5,2.5,value=1.2)#dic['Lens 1 eta'][a1,0,a3])
PA = dic['Lens 1 pa'][a1,0,a3]#pymc.Uniform('Lens 1 pa',0,180.,value=129.)#dic['Lens 1 pa'][a1,0,a3])

SH = dic['extShear'][a1,0,a3]#pymc.Uniform('extShear',-0.3,0.3,value=0.06)#dic['extShear'][a1,0,a3])
SHPA = dic['extShear PA'][a1,0,a3]#pymc.Uniform('extShear PA',-180.,0.,value=-50)#dic['extShear PA'][a1,0,a3])

lens1 = MassModels.PowerLaw('Lens 1',{'x':X,'y':Y,'b':B,'eta':ETA,'q':Q,'pa':PA})
shear = MassModels.ExtShear('shear',{'x':X,'y':Y,'b':SH,'pa':SHPA})
lenses = [lens1,shear]

# set up the source in the gui!
vx,vy=65.1,60.0
SX1 = pymc.Uniform('Source 1 x',vx-5,vx+5,vx)
SY1 = pymc.Uniform('Source 1 y',vy-5,vy+5,vy)
SN1 = pymc.Uniform('Source 1 n',0.5,8.,2.)#gdic['Galaxy 1 n'][ga1,0,ga3])
SRE1 = pymc.Uniform('Source 1 re',0.5,100.,9.1)#gdic['Galaxy 1 re'][ga1,0,ga3])
SPA1 = pymc.Uniform('Source 1 pa',0.,180.,133)#,gdic['Galaxy 1 pa'][ga1,0,ga3])
SQ1 = pymc.Uniform('Source 1 q',0.1,1.,0.59)#gdic['Galaxy 1 q'][ga1,0,ga3])

## source 2
SX2 = pymc.Uniform('Source 2 x',vx-5,vx+5,vx)
SY2 = pymc.Uniform('Source 2 y',vy-5,vy+5,vy)
SN2 = pymc.Uniform('Source 2 n',0.5,8.,5.)#gdic['Galaxy 1 n'][ga1,0,ga3])
SRE2 = pymc.Uniform('Source 2 re',0.5,100.,12.)#gdic['Galaxy 1 re'][ga1,0,ga3])
SPA2 = pymc.Uniform('Source 2 pa',0.,180.,133)#,gdic['Galaxy 1 pa'][ga1,0,ga3])
SQ2 = pymc.Uniform('Source 2 q',0.1,1.,0.59)#gdic['Galaxy 1 q'][ga1,0,ga3])


## source 3

src1 = SBModels.Sersic('Source 1',{'x':SX1,'y':SY1,'n':SN1,'re':SRE1,'pa':SPA1,'q':SQ1})
src2 = SBModels.Sersic('Source 2',{'x':SX2,'y':SY2,'n':SN2,'re':SRE2,'pa':SPA2,'q':SQ2})

srcs = [src1,src2]

#pars = [GX1,GY1,GN1,GRE1,GPA1,GQ1,GX2,GY2,GN2,GRE2,GPA2,GQ2,X,Y,B,Q,ETA,PA,SX1,SY1,SN1,SRE1,SPA1,SQ1,SH,SHPA]
#cov = [0.2,0.2,0.1,0.5,1.,0.1,0.2,0.2,0.1,0.5,1.,0.1,0.1,0.1,0.1,0.05,0.1,1.,0.2,0.2,0.1,0.5,1.,0.1,0.01,1.]

pars = [X,Y,SX1,SY1,SN1,SRE1,SPA1,SQ1,SX2,SY2,SN2,SRE2,SPA2,SQ2]
cov = [0.2,0.2,0.2,0.2,0.1,0.5,1.,0.1,0.2,0.2,0.1,0.5,1.,0.1]

dx,dy=0.,0.
xp,yp = xc+dx,yc+dy
xop,yop = xo+dx,yo+dy

image,sigma,psf = imgs[0], sigs[0],PSFs[0]
imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp[mask2] = gal.pixeval(xin,yin,1./OVRS,csub=21)
        tmp = iT.resamp(tmp,OVRS,True) 
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.getDeflections(lenses,[xin,yin])
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp[mask2] = src.pixeval(x0,y0,1./OVRS,csub=21)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        n +=1
    model[n] = np.ones(model[n-1].shape)
    n+=1
    rhs = (imin/sigin) 
    op = (model/sigin).T 
    fit, chi = optimize.nnls(op,rhs)
    model = (model.T*fit).sum(1)
    resid = (model-imin)/sigin
    lp += -0.5*(resid**2.).sum()
    return lp 
 
  
@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp 

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = numpy.array(cov)
print len(cov), len(pars)

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=60,ntemps=3)
S.sample(500)
outFile = '/data/ljo31/Lens/J1248/'+str(JJ)
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp = result[0]
trace = numpy.array(result[1])
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

jj=0
for jj in range(40):
    S.p0 = trace[-1]
    S.sample(500)

    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()

    result = S.result()
    lp,trace,dic,_ = result

    a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,a3,i]
    print jj
    jj+=1


colours = ['F606W', 'F814W']
#mods = S.blobs
models = []
for i in range(len(imgs)):
    #mod = mods[i]
    #models.append(mod[a1,a2,a3])
    if i == 0:
        dx,dy = 0,0
    else:
        dx = pars[0].value 
        dy = pars[1].value 
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
    model[n] = np.ones(imin.size)
    n+=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
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

