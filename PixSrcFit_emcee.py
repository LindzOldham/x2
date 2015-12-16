from imageSim import SBModels
import indexTricks as iT, pylab as pl, numpy as np, pyfits
import emcee
from scipy import optimize

def func(x,mod,rhs):
    return mod*x - rhs
   
def lnprob(X,xc,yc,rhs):
    if np.any(X<0):
        return -np.inf
    x,y,pa,q,re,n = X
    src = SBModels.Sersic('Source 1',{'x':x,'y':y,'pa':pa,'q':q,'re':re,'n':n})
    model = src.pixeval(xc,yc)
    fit = optimize.leastsq(func,0.1,args=(model,rhs))[0].item()
    lnL = (-0.5*(model*fit -rhs)**2.).sum()
    #print lnL
    return lnL

osrc = np.load('/data/ljo31/Lens/J1605/osrc_F814W.npy')
yc,xc = iT.coords(osrc.shape)
ii = np.where(np.isnan(osrc)==False)
yc,xc = yc[ii],xc[ii]
rhs = osrc[ii].flatten()

# set up emcee
ndim,nwalkers = 6,40 # for instance
p0 = np.zeros((nwalkers, ndim))

p0[:,0] = 200 +np.random.randn(nwalkers)*20 # x
p0[:,1] = 200. +np.random.randn(nwalkers)*20 # y
p0[:,2] = 170. + np.random.randn(nwalkers)*20 # pa
p0[:,3] = 0.8 + np.random.randn(nwalkers)*0.5 # q
p0[:,4] = 25 + np.random.randn(nwalkers)*3 # Re
p0[:,5] = 4. + np.random.randn(nwalkers)*1 # n


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[xc,yc,rhs])
print 'set up sampler'
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print 'sampled'
sampler.run_mcmc(pos, 500, rstate0=state)
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
print("Autocorrelation time:", sampler.get_autocorr_time())
   
chain = sampler.chain
listo = ['x','y','pa','q','Re','n']
for i in range(len(sampler.flatchain[0])):
    pl.figure()
    pl.hist(sampler.flatchain[:,i], 100)
    pl.title("dimension {}".format(i)+" "+str(listo[i]))
    pl.show()
    pl.figure()
    pl.plot(chain[:,:,i].T)
    pl.title("dimension {}".format(i)+" "+str(listo[i]))
    pl.show()
    print listo[i], np.median(sampler.flatchain[:,i]), np.median(p0[:,i])
fchain = sampler.flatchain
flatlnprob = sampler.flatlnprobability

ii = np.argmax(flatlnprob)
x,y,pa,q,Re,n = fchain[ii]
src = SBModels.Sersic('Source 1', {'x':x,'y':y,'pa':pa,'q':q,'re':Re,'n':n})
model = src.pixeval(iT.coords(osrc.shape)[1],iT.coords(osrc.shape)[0])
model2 = src.pixeval(xc,yc)
fit = optimize.leastsq(func,0.1,args=(model2,rhs))[0].item()
pl.figure()
pl.subplot(131)
pl.imshow(osrc,origin='lower',interpolation='nearest',vmin=0,vmax=7)
pl.colorbar()
pl.subplot(132)
pl.imshow(model*fit,origin='lower',interpolation='nearest',vmin=0,vmax=7)
pl.colorbar()
pl.subplot(133)
pl.imshow(osrc-model*fit,origin='lower',interpolation='nearest')
pl.colorbar()

