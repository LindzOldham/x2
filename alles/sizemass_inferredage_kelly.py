import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
from numpy import cos, sin, tan
import cPickle

# from findML_SEDs - inferred ages for source galaxies from photometry and used to calculate masses etc
ages,lvs,iis,ks,mlvs,vs,lis,lks,mlbs  = np.load('/data/ljo31/Lens/Analysis/InferredAges_1src_all.npy')
table = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data
Re, dlumv, dRe, name = table['Re v'], table['lum v hi'], table['Re v hi'], table['name']

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|c|ccccc|}\hline'
print r'name & $R_e (kpc)$ &  $\log(M_{\star})$ & $\log(T/yr)$ & $\Upsilon_v$ \\\hline'
for i in range(len(name)):
    print name[i], '& $', '%.2f'%Re[i], r'\pm', '%.2f'%dRe[i], r'$ & $','%.2f'%(lvs[1,i]+np.log10(mlvs[1,i])), r'\pm', '%.2f'%dlumv[i], '$ & $', '%.2f'%ages[1,i], r'\pm', '%.2f'%(ages[2,i]-ages[1,i]), '$ & $', '%.2f'%mlvs[1,i], r'\pm', '%.2f'%(mlvs[2,i]-mlvs[1,i]), r'$ \\'

ages,logMv,mlv,vi,vk = np.load('/data/ljo31/Lens/LensParams/InferredAges_1src.npy').T
table = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data
Re, dlumv, dRe, name = table['Re v'], table['lum v hi'], table['Re v hi'], table['name']
logRe = np.log10(Re)

x,y = logMv[1:-1], logRe[1:-1]
sxx, syy = dlumv[1:-1], dRe[1:-1]/Re[1:-1]
jj=np.where(y>0) # excluding J1347?
x,y,sxx,syy=x[jj],y[jj],sxx[jj],syy[jj]
sxy, syx = y*0., x*0.
sxx2, syy2 = sxx**2., syy**2.

pars, cov = [], []
pars.append(pymc.Uniform('alpha',-20,10,-10 ))
pars.append(pymc.Uniform('beta',-10,20,1.0 ))
pars.append(pymc.Uniform('sigma',0,0.9,value=0.01))
pars.append(pymc.Uniform('tau',0.001,100,value=1))
pars.append(pymc.Uniform('mu',10,12,value=10.5))
cov += [0.5,0.5,0.01,1.,1.]
optCov = np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    alpha,beta,sigma,tau,mu = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value
    tau2,sigma2,beta2 = tau**2., sigma**2.,beta**2.
    X = x-mu
    Y = y - alpha - beta*mu
    Sxx = sxx + tau2
    Syy = syy + sigma2 + beta2*tau2
    Sxy = sxy + beta*tau2
    Syx = syx + beta*tau2
    Delta = Syy*X**2. + Sxx*Y**2. - X*Y*(Sxy+Syx)
    Sigma =  Sxx*Syy - Sxy*Syx
    pdf = -0.5*Delta/Sigma- 0.5*np.log(Sigma)
    lp = pdf.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=20)
S.sample(4000)
outFile = '/data/ljo31/Lens/Analysis/sizemass_1src_inferredage'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)


pl.figure()
pl.plot(lp)

#for i in range(3):
#    pl.figure()
#    pl.plot(trace[:,:,i])

alpha,beta,sigma,tau,mu = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value
xfit = np.linspace(8,12,20)

# neaten up results - get back to plotting good
burnin=100
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
fits=np.zeros((len(f),xfit.size))
for j in range(0,len(f)):
    alpha,beta,sigma,tau,mu = f[j]
    fits[j] = beta*xfit+alpha
    
lo,med,up = xfit*0.,xfit*0.,xfit*0.
for j in range(xfit.size):
    lo[j],med[j],up[j] = np.percentile(fits[:,j],[16,50,84],axis=0)

los,meds,ups = np.percentile(f,[16,50,84],axis=0)
los,ups=meds-los,ups-meds


pl.figure()
for alpha,beta,sigma,tau,mu in f[np.random.randint(len(f), size=100)]:
    pl.plot(xfit,beta*xfit+alpha,color='k',alpha=0.1)
pl.scatter(x,y)
pl.errorbar(x,y,xerr=sxx,yerr=syy,fmt='o')
pl.axis([10,12,-0.5,2])
pl.xlabel(r'$\log(M/M_{\odot})$')
pl.ylabel(r'$\log(R_e/kpc)$')

yfit=meds[1]*xfit+meds[0]

pl.figure()
pl.plot(xfit,yfit,'Crimson')
pl.fill_between(xfit,yfit,lo,color='LightPink',alpha=0.5)
pl.fill_between(xfit,yfit,up,color='LightPink',alpha=0.5)
pl.scatter(x,y,color='Crimson')
pl.errorbar(x,y,xerr=sxx,yerr=syy,fmt='o',color='Crimson')
pl.xlabel(r'$\log(M/M_{\odot})$')
pl.ylabel(r'$\log(R_e/kpc)$')
pl.title('ages inferred from photometry')

# should also plot vdW's relation!
vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
vdWfit2 = 0.60 - 0.75*(10.+np.log10(5.)) + 0.75*xfit
pl.plot(xfit,vdWfit1,'k:',label='van der Wel+14, z=0.75')
pl.plot(xfit,vdWfit2,'k-.',label='van der Wel+14, z=0.25')
shenfit = np.log10(3.47e-6) + 0.56*xfit
pl.plot(xfit,shenfit,'k-',lw=0.5,label='Shen+03, z= 0')
pl.legend(loc='upper left')
pl.xlim([10,12])
pl.ylim([-0.4,1.9])
# make table of results for this fixed age. Do:
'''
- source gals/1-Sersic/inferred age
- source gals/best-model/inferred age
- may as well do this one, which is bestmodel/source gals/fixed age
'''
print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|ccccccc|}\hline'
print r'Sersic model & age model & $\alpha$ & $\beta$ & $\sigma$ & $\tau$ & $\mu$ \\\hline'
print 'one-component & age=4 Gyr & ', '$','%.2f'%meds[0], '_{-','%.2f'%los[0], '}^{+','%.2f'%ups[0],'}$ & $', '%.2f'%meds[1], '_{-','%.2f'%los[1], '}^{+','%.2f'%ups[1],'}$ & $','%.2f'%meds[2], '_{-','%.2f'%los[2], '}^{+','%.2f'%ups[2],'}$ & $','%.2f'%meds[3], '_{-','%.2f'%los[3], '}^{+','%.2f'%ups[3],'}$ & $','%.2f'%meds[4], '_{-','%.2f'%los[4], '}^{+','%.2f'%ups[4],'}$', r'\\'
print r'\end{tabular}'
print r'\end{table}'
