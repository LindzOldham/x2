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

X = 0 # 211
print X

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
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()

image = py.open('/data/ljo31/Lens/J1619/J1619_nirc2_n_Kp_6x6.fits')[0].data.copy()[200:400,200:400]
sigma = np.ones(image.shape) 

result = np.load('/data/ljo31/Lens/LensModels/J1619_211')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

OVRS = 1
yc,xc = iT.overSample(image.shape,OVRS)
yo,xo = iT.overSample(image.shape,1)
xc,xo,yc,yo=xc*0.2,xo*0.2,yc*0.2,yo*0.2
xc,xo,yc,yo = xc+26,xo+26,yc+29,yo+29

print image.shape

pars = []
cov = []
### 3 PSF components
pars.append(pymc.Uniform('xoffset',-40.,40.,value=0))
pars.append(pymc.Uniform('yoffset',-40.,40.,value=0))
cov += [0.4,0.4]
#psf1
pars.append(pymc.Uniform('sigma 1',0,8,value=1.5))
pars.append(pymc.Uniform('q 1',0,1,value=0.9))
pars.append(pymc.Uniform('pa 1',-180,180,value= 120 ))
pars.append(pymc.Uniform('amp 1',0,1,value=0.5))
cov += [1,0.5,20,0.5]

# psf2
pars.append(pymc.Uniform('sigma 2',0.1,60,value= 15 )) 
pars.append(pymc.Uniform('q 2',0,1,value=0.9))
pars.append(pymc.Uniform('pa 2',-180,180,value= 120 ))
pars.append(pymc.Uniform('amp 2',0,1,value=0.5))
cov += [1,0.5,20,0.5]

# psf3
pars.append(pymc.Uniform('sigma 3',0.1,400,value= 6)) 
pars.append(pymc.Uniform('q 3',0,1,value=0.9))
pars.append(pymc.Uniform('pa 3',-180,180,value= 120 ))
pars.append(pymc.Uniform('amp 3',0,1,value=0.25))
cov += [1,0.5,20,0.5]

psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':pars[2],'q':pars[3],'pa':pars[4],'amp':pars[5]})
psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':pars[6],'q':pars[7],'pa':pars[8],'amp':pars[9]})
psfObj3 = SBObjects.Gauss('psf 3',{'x':0,'y':0,'sigma':pars[10],'q':pars[11],'pa':pars[12],'amp':pars[13]})
psfObjs = [psfObj1,psfObj2,psfObj3]

los = dict([('q',0.05),('pa',-180.),('re',0.1),('n',0.5)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',10.)])
covs = dict([('x',1.),('y',1.),('q',0.1),('pa',10.),('re',10.),('n',1.)])

gals = []
for name in ['Galaxy 1', 'Galaxy 2']:
    p = {}
    if name == 'Galaxy 1':
        for key in 'q','pa','re','n':
            #val = dic[name+' '+key][a1,a2,a3]
            #lo,hi= los[key],his[key]
            #pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            #p[key] = pars[-1]
            #cov.append(covs[key])
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': 
            #val = dic[name+' '+key][a1,a2,a3]
            #lo,hi=val-10,val+10
            #pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            #p[key] = pars[-1]  
            #cov +=[1]
            p[key] = dic[name+' '+key][a1,a2,a3]
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            #val = dic[name+' '+key][a1,a2,a3]
            #lo,hi= los[key],his[key]
            #pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            #p[key] = pars[-1]
            #cov.append(covs[key])
            p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y':
            p[key] = gals[0].pars[key]
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
    if name == 'Source 1':
        for key in 'q','re','n','pa':
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi= los[key],his[key]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(covs[key])
        for key in 'x','y': 
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi=val-10,val+10
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1] + lenses[0].pars[key] 
            cov +=[1]
    srcs.append(SBModels.Sersic(name,p))

xpsf,ypsf = iT.coords((81,81))-40

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    models = []
    dx = pars[0].value
    dy = pars[1].value 
    xp,yp = xc+dx,yc+dy
    psf = xpsf*0.
    for obj in psfObjs:
        obj.setPars()
        psf += obj.pixeval(xpsf,ypsf) / (np.pi*2.*obj.pars['sigma'].value**2.)
    psf = psf/np.sum(psf)
    psf = convolve.convolve(image,psf)[1]
    imin,sigin,xin,yin = image.flatten(),sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xp*0.
        tmp = gal.pixeval(xin,yin,0.2/OVRS,csub=23).reshape(xp.shape)
        tmp = iT.resamp(tmp,OVRS,True) 
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xin,yin],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xp*0.
        tmp = src.pixeval(x0,y0,0.2/OVRS,csub=23).reshape(xp.shape)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n] = -1*np.ones(model[n-1].shape)
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    model = (model.T*fit).sum(1)
    resid = (model-imin)/sigin
    lp = -0.5*(resid**2.).sum()
    return lp 
 
  
@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp #[0]

def resid(p):
    lp = -2*logP.value
    return self.imgs[0].ravel()*0 + lp

optCov = None
if optCov is None:
    optCov = numpy.array(cov)

print len(cov), len(pars)

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=40,ntemps=3)
print 'set up sampler'
S.sample(500)
outFile = '/data/ljo31/Lens/J1619/Kp_211_'+str(X)
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
for jj in range(30):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=40,ntemps=3,initialPars=trace[-1])
    S.sample(500)

    outFile = '/data/ljo31/Lens/J1619/Kp_211_'+str(X)
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()

    result = S.result()
    lp = result[0]

    trace = numpy.array(result[1])
    a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,a3,i]
    print jj
    jj+=1



dx = pars[0].value
dy = pars[1].value 
xp,yp = xc+dx,yc+dy
psf = xpsf*0.
for obj in psfObjs:
    obj.setPars()
    psf += obj.pixeval(xpsf,ypsf) / (np.pi*2.*obj.pars['sigma'].value**2.)
psf = psf/np.sum(psf)
print 'ici',obj.pars['q'].value
psf = convolve.convolve(image,psf)[1]
xp,yp = xc+dx,yc+dy
xop,yop = xo+dy,yo+dy
imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
n = 0
model = np.empty(((len(gals) + len(srcs)),imin.size))
for gal in gals:
    gal.setPars()
    tmp = xc*0.
    tmp = gal.pixeval(xp,yp,0.2/OVRS,csub=23) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
    tmp = src.pixeval(x0,y0,0.2/OVRS,csub=23)
    tmp = iT.resamp(tmp,OVRS,True)
    tmp = convolve.convolve(tmp,psf,False)[0]
    model[n] = tmp.ravel()
    n +=1
rhs = (imin/sigin) # data
op = (model/sigin).T # model matrix
fit, chi = optimize.nnls(op,rhs)
print fit
components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
model = components.sum(0)
NotPlicely(image,model,sigma)
pl.show()
#for i in range(4):
#    pl.figure()
#    pl.imshow(components[i],interpolation='nearest',origin='lower')
#    pl.colorbar()
