import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
from numpy import cos, sin, tan
import cPickle

# need to collate 1- and 2- source models to get ouyr 'best model' for each system!

ages = ['0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','2.750','3.000','3.250','3.500','3.750','4.000','4.250','4.500','4.750','5.000','5.250','5.500','5.750','6.000','7.000','8.000','9.000','10.00','12.00','15.00','20.00']
lumtab = py.open('/data/ljo31/Lens/LensParams/Lumb_ages_2src.fits')[1].data
lumtab2 = py.open('/data/ljo31/Lens/LensParams/Lumb_ages_1src.fits')[1].data
grid = np.zeros((len(ages),9))
for a in range(len(ages)):
    grid[a] = lumtab[ages[a]]
    
grid2 = np.zeros((len(ages),3))
ii=np.where(lumtab2['name']=='J0837')
grid2[:,0] = np.array(lumtab2[ii][0][1:])
ii=np.where(lumtab2['name']=='J0901')
grid2[:,1] = np.array(lumtab2[ii][0][1:])
ii=np.where(lumtab2['name']=='J1218')
grid2[:,2] = np.array(lumtab2[ii][0][1:])

grid = np.column_stack((grid,grid2))

age_array = np.array([0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,2.75,3.0,3.25,3.500,3.75,4.0,4.25,4.500,4.75,5.0,5.25,5.500,5.75,6.000,7.000,8.000,9.000,10.00,12.00,15.00,20.00])
interps = []
for i in range(12):
    interps.append(splrep(age_array,grid[:,i]))


T = 4e9
logT,divT = np.log10(T),T*1e-9
table = py.open('/data/ljo31/Lens/LensParams/Phot_2src.fits')[1].data
table2 = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data

### load up BC03 tables and build interpolators, so we can change the age quickly and update the ML estimate
age_cols,vi, vk = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.2color',unpack=True,usecols=(0,5,7))
age_mls, ml_b, ml_v = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.4color',unpack=True,usecols=(0,4,5))
vimod, vkmod = splrep(age_cols, vi), splrep(age_cols, vk)
mlbmod, mlvmod = splrep(age_mls,ml_b), splrep(age_mls,ml_v)
mlb, mlv = splev(logT,mlbmod), splev(logT,mlvmod)
Re2, dlumb2, dRe2, name2 = table['Re v'], table['lum b hi'], table['Re v hi'], table['name']
###
Re = np.zeros(12)
dlumb, dRe = Re*0.,Re*0.
Re[:9] = Re2
Re[9],Re[10],Re[11] = table2['Re v'][0], table2['Re v'][1], table2['Re v'][5]
dRe[:9] = dRe2
dRe[9],dRe[10],dRe[11] = table2['Re v hi'][0], table2['Re v hi'][1], table2['Re v lo'][5]
dlumb[:9] = dlumb2
dlumb[9],dlumb[10],dlumb[11] = table2['lum b hi'][0], table2['lum b hi'][1], table2['lum b hi'][5]
name = np.concatenate((name2,np.array(['J0837','J0901','J1218'])))
np.save('/data/ljo31/Lens/LensParams/bestmodels_sizemass',np.column_stack((Re,dRe,dlumb)))
ii=np.argsort(name)
name,Re,dRe,dlumb = name[ii], Re[ii],dRe[ii],dlumb[ii]
lumb = [splev(divT,mod).item() for mod in interps]
lumb=np.array(lumb)[ii]
logMb = np.log10(mlb) + lumb
logRe = np.log10(Re)

# set up fit parameters. Now we're fitting the B band -- and skipping J0837 for now
x,y = logMb[1:-1], logRe[1:-1]
sxx, syy = dlumb[1:-1], dRe[1:-1]/Re[1:-1]
sxy, syx = y*0., x*0.
sxx2, syy2 = sxx**2., syy**2.

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|c|ccc|}\hline'
print r'name & $R_e (kpc)$ & $\log(L_B)$ & $\log(M_{\star})$ \\\hline'
for i in range(len(Re)):
    print name[i], '& $', '%.2f'%Re[i], r'\pm', '%.2f'%dRe[i], r'$ & $', '%.2f'%lumb[i], r'\pm', '%.2f'%dlumb[i], r'$ & $', '%.2f'%logMb[i], r'\pm', '%.2f'%dlumb[i], r'$ \\'





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
outFile = '/data/ljo31/Lens/Analysis/sizemass_bestmodels'
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
#pl.figure()
#pl.scatter(x,y,color='SteelBlue')
xfit = np.linspace(8,12,20)
#yfit = beta*xfit + alpha
#pl.plot(xfit,yfit,'k-')

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
pl.title('age = '+'%.2f'%divT+' Gyr')

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
print 'best model & age=4 Gyr & ', '$','%.2f'%meds[0], '_{-','%.2f'%los[0], '}^{+','%.2f'%ups[0],'}$ & $', '%.2f'%meds[1], '_{-','%.2f'%los[1], '}^{+','%.2f'%ups[1],'}$ & $','%.2f'%meds[2], '_{-','%.2f'%los[2], '}^{+','%.2f'%ups[2],'}$ & $','%.2f'%meds[3], '_{-','%.2f'%los[3], '}^{+','%.2f'%ups[3],'}$ & $','%.2f'%meds[4], '_{-','%.2f'%los[4], '}^{+','%.2f'%ups[4],'}$', r'\\'
print r'\end{tabular}'
print r'\end{table}'
