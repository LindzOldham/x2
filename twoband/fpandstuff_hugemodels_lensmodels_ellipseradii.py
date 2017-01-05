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
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_lensgals_huge.fits')[1].data
names = phot['name']

re,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
mu,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)

re1=re*0.
for n in range(names.size):
    name = names[n]
    _,_,re1[n] = np.load('/data/ljo31/Lens/isophotes/'+name+'_211_lens_reff.npy')

#mu -= 10.*np.log10(1.55) # pretty sure we don't dim it, because it cancels out with D_A later on.
sigs = np.load('/data/ljo31/Lens/LensParams/RelogI_lensgals_covariances.npy')
rho = sigs[:,3]
rho=np.delete(rho,3)
rho=np.delete(rho,1)

logI,dlogI = mu*0,mu*0
for ii in range(len(names)):
    band = bands[names[ii]]
    logI[ii] = solarmag.mu_to_logI(mu[ii],band+'_ACS',lz[names[ii]][0])
    dlogI[ii] = logI[ii] - solarmag.mu_to_logI(mu[ii]+dmu[ii],band+'_ACS',lz[names[ii]][0])


xx,yy,zz = np.log10(sigmal), logI.copy(), np.log10(re1)
xx=np.delete(xx,3)
xx=np.delete(xx,1)
dxx,dyy,dzz = dsigmal/sigmal/np.log(10.), dlogI, dre/re1/np.log(10.)
dxx=np.delete(dxx,3)
dxx=np.delete(dxx,1)
#dxx, dyy, dzz = dxx*10.,dyy*10.,dzz*10.
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = -0.99*dyy*dzz,-0.99*dyy*dzz
#np.save('/data/ljo31b/EELs/esi/kinematics/FP_EELs_mu',np.column_stack((xx,yy,zz,dxx,dyy,dzz)))

'''print zz.shape, re.shape
pl.figure()
pl.scatter(1.2*xx-0.75*yy,zz,color='SteelBlue',s=50,alpha=0.5)
pl.scatter(1.2*xx-0.75*yy,np.log10(re),color='Crimson',s=50,alpha=0.5)
pl.figure()
pl.scatter(0.8*xx-0.75*yy,zz,color='SteelBlue',s=50,alpha=0.5)
pl.scatter(0.8*xx-0.75*yy,np.log10(re),color='Crimson',s=50,alpha=0.5)
pl.figure()
pl.scatter(0.35*xx-0.44*yy,zz,color='SteelBlue',s=50,alpha=0.5)
pl.scatter(0.35*xx-0.44*yy,np.log10(re),color='Crimson',s=50,alpha=0.5)
pl.show()'''



pars, cov = [], []
pars.append(pymc.Uniform('a',-2.,3,1.0 ))
pars.append(pymc.Uniform('b',-2.,1,-0.75 ))
pars.append(pymc.Uniform('alpha',-10.,10.,1. ))
pars.append(pymc.Uniform('mu x',-0.5,1.0,0.3 ))
pars.append(pymc.Uniform('mu y',1.,30. ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,1.0 ))
pars.append(pymc.Uniform('rho',-1.,1,-0.5 ))
pars.append(pymc.Uniform('sigma',0,2,0.1 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.1,0.1,0.05]
optCov = np.array(cov)


@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    a,b,alpha,mux,muy,taux,tauy,rho,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value,pars[8].value
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
S.sample(6000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_src_logI_211'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
result = np.load('/data/ljo31b/EELs/FP/inference/FP_src_logI_211')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[2000:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])
print np.amax(lp)

a,b,alpha,mux,muy,taux,tauy = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value

#da, db = a - np.percentile(dic['a'].ravel(),16), b -np.percentile(dic['b'].ravel(),16)

pl.figure()
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')
Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'$')
pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'$')
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ -'+'%.2f'%abs(b)+'$\log I$')
pl.title('EELs')
pl.savefig('/data/ljo31b/EELs/esi/TeX/eelsFPlogi211old.pdf')
pl.show()

