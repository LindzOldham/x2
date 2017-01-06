import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
from numpy import cos, sin, tan
import cPickle

ages = ['0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','2.750','3.000','3.250','3.500','3.750','4.000','4.250','4.500','4.750','5.000','5.250','5.500','5.750','6.000','7.000','8.000','9.000','10.00','12.00','15.00','20.00']
lumtab = py.open('/data/ljo31/Lens/LensParams/Lensinggalaxies_Lumb_ages_1src.fits')[1].data
grid = np.zeros((len(ages),12))
for a in range(len(ages)):
    grid[a] = lumtab[ages[a]]

age_array = np.array([0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,2.75,3.0,3.25,3.500,3.75,4.0,4.25,4.500,4.75,5.0,5.25,5.500,5.75,6.000,7.000,8.000,9.000,10.00,12.00,15.00,20.00])
interps = []
for i in range(12):
    interps.append(splrep(age_array,grid[:,i]))


T = 4e9
logT,divT = np.log10(T),T*1e-9
table = py.open('/data/ljo31/Lens/LensParams/LensinggalaxyPhot_1src.fits')[1].data
### load up BC03 tables and build interpolators, so we can change the age quickly and update the ML estimate
age_cols,vi, vk = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.2color',unpack=True,usecols=(0,5,7))
age_mls, ml_b, ml_v = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.4color',unpack=True,usecols=(0,4,5))
vimod, vkmod = splrep(age_cols, vi), splrep(age_cols, vk)
mlbmod, mlvmod = splrep(age_mls,ml_b), splrep(age_mls,ml_v)
mlb, mlv = splev(logT,mlbmod), splev(logT,mlvmod)
Re, dlumb, dRe, name = table['Re v'], table['lum b hi'], table['Re v hi'], table['name']
lumb = [splev(divT,mod).item() for mod in interps]
logMb = np.log10(mlb) + lumb
logRe = np.log10(Re)

# set up fit parameters. Now we're fitting the B band -- and skipping J0837 for now
x,y = logMb[:-1], logRe[:-1]
sxx, syy = dlumb[:-1], dRe[:-1]/Re[:-1]
jj=np.where(y>0)
x,y,sxx,syy=x[jj],y[jj],sxx[jj],syy[jj]
sxy, syx = y*0., x*0.
sxx2, syy2 = sxx**2., syy**2.

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{|c|ccc|}\hline'
print r'name & $R_e (kpc)$ & $\log(L_B)$ & $\log(M_{\star})$ \\\hline'
for i in range(len(name)):
    print name[i], '& $', '%.2f'%Re[i], r'\pm', '%.2f'%dRe[i], r'$ & $', '%.2f'%lumb[i], r'\pm', '%.2f'%dlumb[i], r'$ & $', '%.2f'%logMb[i], r'\pm', '%.2f'%dlumb[i], r'$ \\'


pars, cov = [], []
pars.append(pymc.Uniform('theta',-np.pi/2., np.pi/2.,value=0.5 ))
pars.append(pymc.Uniform('b',-20,-5,value=-10 ))
pars.append(pymc.Uniform('scatter',0,0.9,value=0.2))
cov += [0.1,0.5,0.05]
optCov = np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    theta,b,scatter = pars[0].value, pars[1].value,pars[2].value
    st,ct = sin(theta), cos(theta)
    Delta = y*ct - x*st - b*ct
    Sigma = sxx2*st**2. + syy2*ct**2. -st*ct*(sxy+syx) + scatter**2.
    pdf = -0.5*Delta**2./Sigma - 0.5*np.log(Sigma)
    lp = pdf.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=20)
S.sample(10000)
outFile = '/data/ljo31/Lens/Analysis/sizemass_bestmodels_lensgals'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

#for i in range(3):
#    pl.figure()
#    pl.plot(trace[:,:,i])



theta,b,scatter = pars[0].value, pars[1].value,pars[2].value
m = tan(theta)
xfit = np.linspace(8,12,20)
yfit = m*xfit + b


burnin=100
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
fits=np.zeros((len(f),xfit.size))
for j in range(0,len(f)):
    theta,b,scatter = f[j]
    m=tan(theta)
    fits[j] = m*xfit+b
    
lo,med,up = yfit*0.,yfit*0.,yfit*0.
for j in range(xfit.size):
    lo[j],med[j],up[j] = np.percentile(fits[:,j],[16,50,84],axis=0)

los,meds,ups = np.percentile(f,[16,50,84],axis=0)
los,ups=meds-los,ups-meds


'''pl.figure()
for theta,b,scatter in f[np.random.randint(len(f), size=100)]:
    m=tan(theta)
    pl.plot(xfit,m*xfit+b,color='k',alpha=0.1)
pl.scatter(x,y)
pl.errorbar(x,y,xerr=sxx,yerr=syy,fmt='o')
pl.axis([10,12,-0.5,2])
pl.xlabel(r'$\log(M/M_{\odot})$')
pl.ylabel(r'$\log(R_e/kpc)$')
'''
yfit=tan(meds[0])*xfit+meds[1]

#pl.figure()
pl.plot(xfit,yfit,'Navy')
pl.fill_between(xfit,yfit,lo,color='LightBlue',alpha=0.5)
pl.fill_between(xfit,yfit,up,color='LightBlue',alpha=0.5)
pl.scatter(x,y,color='Navy')
pl.errorbar(x,y,xerr=sxx,yerr=syy,fmt='o',color='Navy')
pl.xlabel(r'$\log(M/M_{\odot})$')
pl.ylabel(r'$\log(R_e/kpc)$')
pl.title('age = '+'%.2f'%divT+' Gyr')

# should also plot vdW's relation!
#vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
#vdWfit2 = 0.60 - 0.75*(10.+np.log10(5.)) + 0.75*xfit
#pl.plot(xfit,vdWfit1,'k:',label='van der Wel+14, z=0.75')
#pl.plot(xfit,vdWfit2,'k-.',label='van der Wel+14, z=0.25')
shenfit = np.log10(3.47e-6) + 0.56*xfit
pl.plot(xfit,shenfit,'k-',lw=0.5,label='Shen+03, z= 0')
pl.legend(loc='upper left')
pl.xlim([10.5,12])
pl.ylim([0,2])
