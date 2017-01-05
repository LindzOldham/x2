import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances
from SampleOpt import AMAOpt
from astLib import astCalc


fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results_0.30_source_indous_vdfit_jul2016.npy')
l,m,u = fp
d = np.mean((l,u),axis=0)
dvl,dvs,dsigmal,dsigmas = d.T
vl,vs,sigmal,sigmas = m.T
dsigmas = sigmas*0.05
sigmas /= 100.
sigmal /= 100.
dsigmas /= 100.
dsigmal /= 100.
sigmas = np.loadtxt('/data/ljo31b/EELs/phys_models/models/noDM2.dat')/100.
# remove J1248 as we don't have photometry
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_lens_vdfit.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_1.00_lens_vdfit.npy')[()]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_huge_new_new.fits')[1].data
names = phot['name']

re,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
mu,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)
scales = np.array([astCalc.da(sz[name][0])*1e3*np.pi/180./3600. for name in names])
mag = mu - 2.5*np.log10(2.*np.pi*re**2./scales**2.)

sigs = np.load('/data/ljo31/Lens/LensParams/ReMu_covariances.npy')
rho = sigs[:,3]

# convert mu to logI!
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
logI,dlogI = mu*0,mu*0
logL = mu*0
logLR = mu*0.

for ii in range(len(names)):
    band = bands[names[ii]]
    logL[ii] = solarmag.mag_to_logL(mag[ii],band+'_ACS',sz[names[ii]][0]) - 8.
    logLR[ii] =  np.log10(2.*np.pi*re[ii]**2.)
    logI[ii] =  logL[ii] - logLR[ii]

dlogI = 0.4*dmu
print logI


xx,yy,zz = np.log10(sigmas), logI, np.log10(re)
xo,yo,zo = xx.copy(),yy.copy(),zz.copy()
dxx,dyy,dzz = dsigmas[:-1]/sigmas/np.log(10.), dlogI, dre/re/np.log(10.)
dxo,dyo,dzo = dxx.copy(),dyy.copy(),dzz.copy()
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
syz,szy = rho*dyy*dzz,rho*dyy*dzz

pl.figure()
pl.scatter(1.57*xx-0.7*yy[:-3],zz[:-3])
pl.show()

I=-1
names = np.delete(names,I)
yy,zz,syy,szz = np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
I=-1
yy,zz,syy,szz = np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
names = np.delete(names,I)
I = -1
yy,zz,syy,szz = np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
names = np.delete(names,I)
print names


pars, cov = [], []
pars.append(pymc.Uniform('a',-10.,10,1.0 ))
pars.append(pymc.Uniform('b',-5.,5,-0.8 ))
pars.append(pymc.Uniform('alpha',-20.,20.,-6. ))
pars.append(pymc.Uniform('mu x',-2,2,0.3 ))
pars.append(pymc.Uniform('mu y',-10.,30. ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,1.0 ))
pars.append(pymc.Uniform('rho',-1.,1 ))
pars.append(pymc.Uniform('sigma',0,10,0.2 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.1,0.1,0.1]
optCov = np.array(cov)


@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    a,b,alpha,mux,muy,taux,tauy,rho,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value, pars[8].value
    tauxy,tauyx = rho*taux*tauy, rho*taux*tauy
    taux2,tauy2,sigma2,a2,b2 = taux**2.,tauy**2.,sigma**2.,a**2.,b**2.
    X = zz - alpha - a*mux - b*muy
    Y = xx - mux
    Z = yy - muy
    Sxx = taux2*a2 + a*b*(tauxy+tauyx) + tauy2*b2 + sigma2 + szz #
    Syy = taux2 + sxx 
    Szz = tauy2 + syy 
    Sxy = a*taux2 + b*tauyx + sxz 
    Sxz = a*tauxy + b*tauy2 + syz 
    Syx = taux2*a + tauxy*b + sxz 
    Szx = tauy2*b + tauyx*a +  syz 
    Syz = tauxy + sxy 
    Szy = tauyx + syx 
    resid = 0
    args = np.zeros(X.size)
    for ii in range(X.size):
        V = np.matrix([[Sxx[ii], Sxy, Sxz[ii]],[Syx,Syy[ii],Syz],[Szx[ii],Szy,Szz[ii]]])
        Vinv = V.I
        Vdet = np.linalg.det(V)
        ZZ = np.matrix([[X[ii],Y[ii],Z[ii]]]).T
        args[ii] = -0.5*np.dot(ZZ.T,np.dot(Vinv,ZZ)) - 0.5*np.log(np.abs(Vdet))
        resid += args[ii]
    return resid

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

# optimise first!

SS = AMAOpt(pars,[likelihood],[logP],cov=cov)
SS.sample(4000)
lp,trace,det = SS.result()
#pl.figure()
#pl.plot(lp)
#pl.show() 
print 'results from optimisation:'
for i in range(len(pars)):
    pars[i].value = trace[-1,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

print 'new!'
S = myEmcee.Emcee(pars+[likelihood],cov=optCov/5.,nthreads=8,nwalkers=28)
S.sample(5000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_10_models_logI'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[2000:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

a,b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value
da, db = np.percentile(dic['a'][2000:].ravel(),84)-a, np.percentile(dic['b'][2000:].ravel(),84)-b

yerr = szz**0.5
xerr = (a**2.*sxx + b**2*syy)**0.5
pl.figure()

Zline = np.linspace(min(zz)-0.5,max(zz)+0.5,10)
pl.plot(Zline,Zline,color='k')
pl.fill_between(Zline,Zline,Zline+sigma,color='LightGray')
pl.fill_between(Zline,Zline,Zline-sigma,color='LightGray')

pl.scatter(a*xx+b*yy+alpha,zz,color='k')
pl.errorbar(a*xx+b*yy+alpha,zz,xerr=xerr,yerr=yerr,color='k',fmt='o')

#pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
#pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ '+'%.2f'%b+'$\log I$')
pl.title('EELs')
#pl.scatter(a*xo[-1]+b*yo[-1]+alpha,zo[-1],color='Crimson')
#pl.scatter(a*xo[3]+b*yo[3]+alpha,zo[3],color='Crimson')

#pl.savefig('/data/ljo31b/EELs/esi/TeX/eelsFPmu.pdf')
pl.axis([0,1.1,0,1.1])
pl.show()




