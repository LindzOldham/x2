import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee

''' code to model the star in DES as a Gaussian. Cameron's job is to replicate this for a Moffat profile '''

image = py.open('/data/desardata/Y1A1/3070265353/3070265353_i.fits')[1].data.copy()[0:12,2:16]

# define parameters of the Gaussian to be inferred
pars, cov = [], []
pars.append(pymc.Uniform('x',0,10,value=6.5))
pars.append(pymc.Uniform('y',0,10,value=7))
pars.append(pymc.Uniform('sigma',0,10,value=1))
pars.append(pymc.Uniform('q',0.1,1,value=0.9))
pars.append(pymc.Uniform('pa',-180,180,value= 0 ))
pars.append(pymc.Uniform('amp',0,3000,value= 2000 ))
pars.append(pymc.Uniform('background',-100,100,value=0 ))
cov += [1,1,1,0.1,50,100,10]
# set up PSF object and the grid that we'll evaluate the PSF on
psfObj = SBObjects.Gauss('psf',{'x':pars[0],'y':pars[1],'sigma':pars[2],'q':pars[3],'pa':pars[4],'amp':pars[5]})
psfObj.setPars()
ypsf,xpsf = iT.coords(image.shape)

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]

# define likelihood 
@pymc.deterministic
def logP(value=0.,p=pars):
    psfObj.setPars()
    psf = psfObj.pixeval(xpsf,ypsf)
    psf += np.ones(psf.shape)*pars[-1].value
    resid = image-psf
    lp = -0.5*(resid**2.).sum()
    return lp
  
# define the function that CALLS the likelihood (pymc is weird)
@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp 

optCov = numpy.array(cov)

# set up emcee and run the mcmc chain
S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=30)
print 'set up sampler'
S.sample(10000)
result = S.result()
lp,trace,dic,_ = result
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)

for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

# evaluate maximum-likelihood model and compare with data. Note the final plot shows the log likelihood, which has stopped increasing by the end of the chain -- this shows it has converged
psfObj.setPars()
psf = psfObj.pixeval(xpsf,ypsf)
psf += np.ones(psf.shape)*pars[-1].value
pl.figure()
pl.imshow(image,interpolation='nearest',origin='lower',cmap='afmhot')
pl.colorbar()
pl.title('image')
pl.figure()
pl.imshow(psf,interpolation='nearest',origin='lower',cmap='afmhot')
pl.colorbar()
pl.title('model')
pl.figure()
pl.imshow((image-psf)/image,interpolation='nearest',origin='lower',cmap='afmhot')
pl.colorbar()
pl.title('residuals')

pl.figure()
pl.plot(lp[1000:])

# have also converged on final values for each of the parameters...set plottrace = True to plot them!
plottrace = False
if plottrace:
    for i in range(trace.shape[-1]):
        pl.figure()
        pl.plot(trace[:,:,i])
