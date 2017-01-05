import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
from numpy import cos, sin, tan
import cPickle
from tools.EllipsePlot import *

# from findML_SEDs - inferred ages for source galaxies from photometry and used to calculate masses etc
model_vi,masses,model_vk,ages = np.load('/data/ljo31/Lens/Analysis/InferredAges_1src_all_new.npy')
filename = '/data/ljo31b/EELs/inference/masses.npy'
fi = open(filename)
for i in range(12):
    masses[:,i] = np.load(fi)[3:]

# check order there!!!
vi,vk,dvi,dvk,v,i,k,Re,dRe = np.load('/data/ljo31/Lens/LensParams/colours_1src.npy')
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
logRe = np.log10(Re)
logM = masses[0]
dlogM = np.mean((masses[2],masses[1]),axis=0)
dlogRe = dRe/Re
rho = np.load('/data/ljo31/Lens/LensParams/ReMass_covariances.npy')[:,0]
np.save('/data/ljo31/Lens/LensParams/ReMass_1src_allparams',np.column_stack((logRe,logM,dlogRe,dlogM,rho)))

print 'this table is meaningless'
print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|c|ccccccc|}\hline'
print r'name & $R_e (kpc)$ &  $\log(M_{\star})$ & $\log(T/yr)$ & $v-i$ & $v-i$ (mod) & $v-k$ & $v-k$ (mod)  \\\hline'
for i in range(len(name)):
    print name[i], '& $', '%.2f'%Re[i], r'\pm', '%.2f'%dRe[i], r'$ & $','%.2f'%masses[0,i], r'\pm', '%.2f'%(masses[2,i]), '$ & $', '%.2f'%ages[1,i], r'\pm', '%.2f'%(ages[2,i]-ages[1,i]), '$ & $','%.2f'%vi[i], r'\pm', '%.2f'%dvi[i],'$ & $','%.2f'%model_vi[1,i], r'\pm', '%.2f'%(model_vi[2,i]-model_vi[1,i]),'$ & $', '%.2f'%vk[i], r'\pm', '%.2f'%dvk[i],'$ & $', '%.2f'%model_vk[1,i], r'\pm', '%.2f'%(model_vk[2,i]-model_vk[1,i]),r'$ \\'

x,y = logM, logRe 
sxx, syy = dlogM, dlogRe
sxx2, syy2 = sxx**2., syy**2.
sxy,syx = rho*sxx*syy, rho*syy*sxx

pars, cov = [], []
pars.append(pymc.Uniform('alpha',-20,10,-10 ))
pars.append(pymc.Uniform('beta',-10,20,1.0 ))
pars.append(pymc.Uniform('sigma',0,0.9,value=0.01))
pars.append(pymc.Uniform('tau',0.001,100,value=1))
pars.append(pymc.Uniform('mu',10,12,value=11))
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
outFile = '/data/ljo31/Lens/Analysis/sizemass_1src_inferredage_allparams'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
#for i in range(len(pars)):
#    pars[i].value = trace[a1,a2,i]
#    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)
ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
for i in range(len(pars)):
    pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)


pl.figure()
pl.plot(lp)

#for i in range(3):
#    pl.figure()
#    pl.plot(trace[:,:,i])

alpha,beta,sigma,tau,mu = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value
xfit = np.linspace(8,14,20)
pl.figure()
pl.scatter(x,y,color='SteelBlue')
pl.plot(xfit,xfit*beta + alpha,'SteelBlue',label='EELs')
#pl.errorbar(x,y,xerr=sxx,yerr=syy,fmt='o',color='SteelBlue')
plot_ellipses(x,y,sxx,syy,rho,'SteelBlue')
pl.xlabel(r'$\log(M/M_{\odot})$')
pl.ylabel(r'$\log(R_e/kpc)$')
vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
vdWfit2 = 0.60 - 0.75*(10.+np.log10(5.)) + 0.75*xfit
shenfit = np.log10(3.47e-6) + 0.56*xfit
pl.plot(xfit,vdWfit1,'k:',label='van der Wel+14, z=0.75')
pl.plot(xfit,vdWfit2,'k-.',label='van der Wel+14, z=0.25')
pl.plot(xfit,shenfit,'k-',lw=0.5,label='Shen+03, z= 0')
pl.legend(loc='upper left')
pl.xlim([10.5,12.5])
pl.ylim([-0.4,1.9])
pl.show()

# make a table
burnin=200
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
fits=np.zeros((len(f),xfit.size))
for j in range(0,len(f)):
    alpha,beta,sigma,tau,mu = f[j]
    fits[j] = beta*xfit+alpha

los,meds,ups = np.percentile(f,[16,50,84],axis=0)
los,ups=meds-los,ups-meds
print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|ccccccc|}\hline'
print r'Sersic model & age model & $\alpha$ & $\beta$ & $\sigma$ & $\tau$ & $\mu$ \\\hline'
print 'one-component & from photometry & ', '$','%.2f'%meds[0], '_{-','%.2f'%los[0], '}^{+','%.2f'%ups[0],'}$ & $', '%.2f'%meds[1], '_{-','%.2f'%los[1], '}^{+','%.2f'%ups[1],'}$ & $','%.2f'%meds[2], '_{-','%.2f'%los[2], '}^{+','%.2f'%ups[2],'}$ & $','%.2f'%meds[3], '_{-','%.2f'%los[3], '}^{+','%.2f'%ups[3],'}$ & $','%.2f'%meds[4], '_{-','%.2f'%los[4], '}^{+','%.2f'%ups[4],'}$', r'\\'
print r'\end{tabular}'
print r'\end{table}'

# make plots with uncertainties
yfit=meds[1]*xfit+meds[0]
lo,med,up = xfit*0.,xfit*0.,xfit*0.
for j in range(xfit.size):
    lo[j],med[j],up[j] = np.percentile(fits[:,j],[16,50,84],axis=0)

pl.figure()
pl.plot(xfit,yfit,'SteelBlue')
pl.fill_between(xfit,yfit,lo,color='LightBlue',alpha=0.5)
pl.fill_between(xfit,yfit,up,color='LightBlue',alpha=0.5)
pl.scatter(x,y,color='SteelBlue')
#pl.errorbar(x,y,xerr=sxx,yerr=syy,fmt='o',color='SteelBlue')
plot_ellipses(x,y,sxx,syy,rho,'SteelBlue')
pl.xlabel(r'$\log(M_{\star}/M_{\odot})$')
pl.ylabel(r'$\log(R_e/kpc)$')
vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
vdWfit2 = 0.60 - 0.75*(10.+np.log10(5.)) + 0.75*xfit
shenfit = np.log10(3.47e-6) + 0.56*xfit
pl.plot(xfit,vdWfit1,'k:',label='van der Wel+14, z=0.75')
pl.plot(xfit,vdWfit2,'k-.',label='van der Wel+14, z=0.25')
pl.plot(xfit,shenfit,'k-',lw=0.5,label='Shen+03, z= 0')
pl.legend(loc='upper left')
pl.xlim([10,12])
pl.ylim([-0.4,1.9])
pl.show()
# run loads of realisations of the EELs models to get covariance




np.save('/data/ljo31/Lens/Analysis/sizemass_1src_fit_allparams',np.column_stack((xfit,yfit,lo,up)))

