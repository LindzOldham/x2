from imageSim import SBModels
import indexTricks as iT, pylab as pl, numpy as np, pyfits
import emcee
from scipy import optimize

   
def lnprob(X,xc,yc,rhs):
    x,y,pa,q,re,n,pa2,q2,re2,n2 = X
    if np.any(np.array([x,y,q,q2,n,n2,re,re2])<0) or n<0.1 or n2<0.1 or n>6 or n2>6 or re>300 or re2>300 or q>1 or q2>1 or pa>200 or pa2>200 or pa<-180 or pa2<-180:
        return -np.inf
    src = SBModels.Sersic('Source 1',{'x':x,'y':y,'pa':pa,'q':q,'re':re,'n':n})
    src2 = SBModels.Sersic('Source 2',{'x':x,'y':y,'pa':pa2,'q':q2,'re':re2,'n':n2})
    model1 = src.pixeval(xc,yc) 
    model2 = src2.pixeval(xc,yc)
    op = np.column_stack((model1,model2))
    fit = optimize.nnls(op,rhs)[0]
    lnL = (-0.5*(model1*fit[0] + model2*fit[1] - rhs)**2.).sum()
    #lnL = (-0.5*(np.dot(op,fit) -rhs)**2.).sum()
    return lnL

osrc = np.load('/data/ljo31/Lens/J1605/osrc_F814W.npy')
yc,xc = iT.coords(osrc.shape)
ii = np.where(np.isnan(osrc)==False)
yc,xc = yc[ii],xc[ii]
rhs = osrc[ii].flatten()

# set up emcee
ndim,nwalkers = 10,60 # for instance
p0 = np.zeros((nwalkers, ndim))

p0[:,0] = 220 +np.random.randn(nwalkers)*20 # x
p0[:,1] = 200. +np.random.randn(nwalkers)*20 # y
p0[:,2] = 0. + np.random.randn(nwalkers)*20 # pa
p0[:,3] = 0.7 + np.random.randn(nwalkers)*0.1 # q
p0[:,4] = 220 + np.random.randn(nwalkers)*20 # Re # 120
p0[:,5] = 4. + np.random.randn(nwalkers)*1 # n
p0[:,6] = 150. + np.random.randn(nwalkers)*20 # pa
p0[:,7] = 0.7 + np.random.randn(nwalkers)*0.1 # q
p0[:,8] = 20 + np.random.randn(nwalkers)*3 # Re
p0[:,9] = 2. + np.random.randn(nwalkers)*1 # n

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[xc,yc,rhs])
print 'set up sampler'
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print 'sampled'
sampler.run_mcmc(pos, 800, rstate0=state)
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
print("Autocorrelation time:", sampler.get_autocorr_time())
   
chain = sampler.chain
listo = ['x','y','pa','q','Re','n','pa2','q2','Re2','n2']
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
x,y,pa,q,re,n,pa2,q2,re2,n2 = fchain[ii]
src = SBModels.Sersic('Source 1', {'x':x,'y':y,'pa':pa,'q':q,'re':re,'n':n})
src2 = SBModels.Sersic('Source 2',{'x':x,'y':y,'pa':pa2,'q':q2,'re':re2,'n':n2})
model1 = src.pixeval(xc,yc) 
model2 = src2.pixeval(xc,yc)
op = np.column_stack((model1,model2))
fit = optimize.nnls(op,rhs)[0]

model = fit[0] * src.pixeval(iT.coords(osrc.shape)[1],iT.coords(osrc.shape)[0]) + fit[1] * src2.pixeval(iT.coords(osrc.shape)[1],iT.coords(osrc.shape)[0])

pl.figure()
pl.subplot(131)
pl.imshow(osrc,origin='lower',interpolation='nearest',vmin=0,vmax=7)
pl.colorbar()
pl.subplot(132)
pl.imshow(model,origin='lower',interpolation='nearest',vmin=0,vmax=7)
pl.colorbar()
pl.subplot(133)
pl.imshow(osrc-model,origin='lower',interpolation='nearest')
pl.colorbar()

print 'fit', fit
