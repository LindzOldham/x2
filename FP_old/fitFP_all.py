import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances

fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results.npy')
l,m,u = fp
d = np.mean((l,u),axis=0)
dvl,dvs,dsigmal,dsigmas = d.T
vl,vs,sigmal,sigmas = m.T
sigmas /= 100.
sigmal /= 100.
dsigmas /= 100.
dsigmal /= 100.
# remove J1248 as we don't have photometry
sigmas,sigmal,dsigmas,dsigmal = np.delete(sigmas,6), np.delete(sigmal,6),np.delete(dsigmas,6),np.delete(dsigmal,6)
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
names = phot['name']

re,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
mu,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)

#mu -= 10.*np.log10(1.55) # pretty sure we don't dim it, because it cancels out with D_A later on.
sigs = np.load('/data/ljo31/Lens/LensParams/RelogI_covariances.npy')
rho = sigs[:,3]

logI,dlogI = mu*0.,mu*0.
for n in range(len(mu)):
    logI[n] = solarmag.mu_to_logI(mu[n],'F606W_ACS',sz[names[n]][0])
    dlogI[n] = logI[n] - solarmag.mu_to_logI(mu[n]+dmu[n],'F606W_ACS',sz[names[n]][0])


xx,yy,zz = np.log10(sigmas), logI.copy(), np.log10(re)
dxx,dyy,dzz = dsigmas/sigmas/np.log(10.), dlogI, dre/re/np.log(10.)
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = rho*dyy*dzz,rho*dyy*dzz
np.save('/data/ljo31b/EELs/esi/kinematics/FP_EELs',np.column_stack((xx,yy,zz,dxx,dyy,dzz)))
# need to get cvariance? Have this between re and mu somewhere already?

pl.figure()
pl.hist(xx,30)
pl.figure()
pl.hist(yy,30)
pl.figure()
pl.hist(zz,30)
pl.show()

pars, cov = [], []
pars.append(pymc.Uniform('a',0.,2,1.1 ))
pars.append(pymc.Uniform('b',-2.,2.,-0.7 ))
pars.append(pymc.Uniform('alpha',-20.,20.,0. ))
pars.append(pymc.Uniform('mu x',-0.5,1.0 ))
pars.append(pymc.Uniform('mu y',1.0,2.5 ))
pars.append(pymc.Uniform('tau x',0,3,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,0.1 ))
pars.append(pymc.Uniform('rho',-1.,1 ))
pars.append(pymc.Uniform('sigma',0,10,0.1 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.05,0.1,0.05]
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


S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=8,nwalkers=28)
S.sample(5000)
outFile = '/data/ljo31b/EELs/FP/inference/FP'
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
pl.plot(lp[200:])

a,b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value

pl.figure()
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')
Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'a = '+'%.2f'%a)
pl.figtext(0.2,0.75,'b = '+'%.2f'%b)
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ -'+'%.2f'%abs(b)+'$\log I$')
pl.show()

for key in dic.keys():
    pl.figure()
    pl.plot(dic[key])
    pl.title(key)
pl.show()

