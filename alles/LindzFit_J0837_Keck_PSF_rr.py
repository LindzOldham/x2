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
X=0 - TO RUN. 2 PSF components, one big one small
'''
X=0# with flatr noisemap
X=1 # with new proper noisemap!
X='212_1_ctd'
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
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0,vmax=2) #,vmin=vmin,vmax=vmax)
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

result = np.load('/data/ljo31/Lens/LensModels/J0837_211')
lp= result[0]
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
trace = result[1]
dic = result[2]

kresult = np.load('/data/ljo31/Lens/J0837/KeckPSF_212_1')
klp= kresult[0]
ka2=0
ka1,ka3 = numpy.unravel_index(klp[:,0].argmax(),klp[:,0].shape)
ktrace = kresult[1]
kdic = kresult[2]
xoffset,yoffset,sig1,q1,pa1,amp1,sig2,q2,pa2,amp2,sig3,q3,pa3,amp3,sig4,q4,pa4,amp4 = ktrace[ka1,ka2,ka3,:18]


OVRS = 1
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

pars = []
cov = []
### first parameters need to be the offsets
pars.append(pymc.Uniform('xoffset',-10.,10.,value=xoffset))
pars.append(pymc.Uniform('yoffset',-10.,10.,value=yoffset))
cov += [0.4,0.4]
#psf1
pars.append(pymc.Uniform('sigma 1',0,8,value=sig1))
pars.append(pymc.Uniform('q 1',0,1,value=q1))
pars.append(pymc.Uniform('pa 1',-180,180,value= pa1 ))
pars.append(pymc.Uniform('amp 1',0,1,value=amp1))
cov += [1,0.5,50,0.5]

# psf2
pars.append(pymc.Uniform('sigma 2',0.1,60,value=sig2)) 
pars.append(pymc.Uniform('q 2',0,1,value=q2))
pars.append(pymc.Uniform('pa 2',-180,180,value= pa2 ))
pars.append(pymc.Uniform('amp 2',0,1,value=amp2))
cov += [1,0.5,50,0.5]

# psf3
pars.append(pymc.Uniform('sigma 3',0.1,400,value=sig3 )) 
pars.append(pymc.Uniform('q 3',0,1,value=q3))
pars.append(pymc.Uniform('pa 3',-180,180,value= pa3 ))
pars.append(pymc.Uniform('amp 3',0,1,value=amp3))
cov += [1,0.5,50,0.5]

# psf4
pars.append(pymc.Uniform('sigma 4',0.1,400,value= sig4 )) 
pars.append(pymc.Uniform('q 4',0,1,value=q4))
pars.append(pymc.Uniform('pa 4',-180,180,value= pa4 ))
pars.append(pymc.Uniform('amp 4',0,1,value=amp4))
cov += [1,0.5,50,0.5]


psfObj1 = SBObjects.Gauss('psf 1',{'x':0,'y':0,'sigma':pars[2],'q':pars[3],'pa':pars[4],'amp':pars[5]})
psfObj2 = SBObjects.Gauss('psf 2',{'x':0,'y':0,'sigma':pars[6],'q':pars[7],'pa':pars[8],'amp':pars[9]})
psfObj3 = SBObjects.Gauss('psf 3',{'x':0,'y':0,'sigma':pars[10],'q':pars[11],'pa':pars[12],'amp':pars[13]})
psfObj4 = SBObjects.Gauss('psf 4',{'x':0,'y':0,'sigma':pars[14],'q':pars[15],'pa':pars[16],'amp':pars[17]})
psfObjs = [psfObj1,psfObj2,psfObj3,psfObj4]

xpsf,ypsf = iT.coords((101,101))-50

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
for name in ['Source 2', 'Source 1']:
    p = {}
    if name == 'Source 2':
        print name
        key = 'q'
        pars.append(pymc.Uniform('%s %s'%(name,key),0.05,1,value=kdic[name+' '+key][ka1,ka2,ka3] ))
        p[key] = pars[-1]
        key = 'n'
        pars.append(pymc.Uniform('%s %s'%(name,key),0.1,8,value=kdic[name+' '+key][ka1,ka2,ka3] ))
        p[key] = pars[-1]
        key = 're'
        pars.append(pymc.Uniform('%s %s'%(name,key),0.1,100,value=kdic[name+' '+key][ka1,ka2,ka3] ))
        p[key] = pars[-1]
        key = 'pa'
        pars.append(pymc.Uniform('%s %s'%(name,key),-180,180,value=kdic[name+' '+key][ka1,ka2,ka3] ))
        p[key] = pars[-1]
        cov += [0.1,0.1,1,1]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            #p[key] = dic[name+' '+key][a1,a2,a3]#+lenses[0].pars[key]
            val = kdic[name+' '+key][ka1,ka2,ka3]
            lo,hi=val-20,val+20
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1] #+ lenses[0].pars[key] 
            cov +=[5]
    elif name == 'Source 1':
        print name
        for key in 'q','re','n','pa':
           p[key] = dic[name+' '+key][a1,a2,a3]
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            #p[key] = dic[name+' '+key][a1,a2,a3]+lenses[0].pars[key]
            val = kdic[name+' '+key][ka1,ka2,ka3]
            lo,hi=val-20,val+20
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1] + lenses[0].pars[key] 
            cov +=[5]
    srcs.append(SBBModels.Sersic(name,p))

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
    if obj.pars['amp'].value<0:
        return -1e10
    psf = psf/np.sum(psf)
    psf = convolve.convolve(image,psf)[1]
    imin,sigin,xin,yin = image[mask], sigma[mask],xp[mask2],yp[mask2]
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        gal.setPars()
        tmp = xc*0.
        tmp[mask2] = gal.pixeval(xin,yin,0.2,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
        tmp[mask2] = src.pixeval(x0,y0,0.2,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp[mask].ravel()
        if src.name== 'Source 2':
            model[n] *= -1
        n +=1
    model[n] = np.ones(model[n-1].shape)
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

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=52,ntemps=3,initialPars=ktrace[-1])
S.sample(500)
outFile = '/data/ljo31/Lens/J0837/KeckPSF_'+str(X)
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
for jj in range(20):
    S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=52,ntemps=3,initialPars=trace[-1])
    S.sample(500)

    outFile = '/data/ljo31/Lens/J0837/KeckPSF_'+str(X)
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
    psf += obj.pixeval(xpsf,ypsf) #/ (np.pi*2.*obj.pars['sigma'].value**2.)
psf = psf/np.sum(psf)
psf = convolve.convolve(image,psf)[1]
xp,yp = xc+dx,yc+dy
xop,yop = xo+dy,yo+dy
imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
n = 0
model = np.empty(((len(gals) + len(srcs)+1),imin.size))
for gal in gals:
    gal.setPars()
    tmp = xc*0.
    tmp = gal.pixeval(xp,yp,0.2,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
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
    tmp = src.pixeval(x0,y0,0.2,csub=1)
    tmp = iT.resamp(tmp,OVRS,True)
    tmp = convolve.convolve(tmp,psf,False)[0]
    model[n] = tmp.ravel()
    if src.name == 'Source 2':
        model[n] *= -1
    n +=1
model[n]=np.ones(model[n].shape)
n+=1
rhs = (imin/sigin) # data
op = (model/sigin).T # model matrix
fit, chi = optimize.nnls(op,rhs)
components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
model = components.sum(0)
NotPlicely(image,model,sigma)
#for i in np.arange(2,4):
#    pl.figure()
#    pl.imshow(components[i],interpolation='nearest',origin='lower')
#    pl.colorbar()
print fit

pl.figure()
pl.plot(klp[:,0])
