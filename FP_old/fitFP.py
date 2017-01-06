import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

names = ['J0837','J0901','J0913','J1125','J1144','J1218']
vdl,vd,vdu = np.load('/data/ljo31b/EELs/esi/kinematics/inference/night1.npy')
dvd = np.mean((vdl,vdu),axis=0)
vlens,siglens,vsrc,sigsrc = vd.T
dvlens,dsiglens,dvsrc,dsigsrc = dvd.T
sz = dict([('J0837',0.6411),('J0901',0.586),('J0913',0.539),('J1125',0.689),('J1144',0.706),('J1218',0.6009),('J1248',0.528),('J1323',0.4641),('J1347',0.63),('J1446',0.585),('J1605',0.542),('J1606',0.6549),('J1619',0.6137),('J2228',0.4370)])

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
allnames = phot['name']

re,rel,reu = phot['Re i'], phot['Re i lo'], phot['Re i hi']
dre = np.mean((rel,reu),axis=0)

mu,mul,muu = phot['mu i'], phot['mu i lo'], phot['mu i hi']
dmu = np.mean((mul,muu),axis=0)

re,dre,mu,dmu = re[:6],dre[:6],mu[:6],dmu[:6]
mu -= 10.*np.log10(1.55)

sigsrc /= 100.
dsigsrc /= 100.

logI,dlogI = mu*0.,mu*0.
for n in range(len(names)):
    logI[n] = solarmag.mu_to_logI(mu[i],'F814W_ACS',sz[names[n]])
    dlogI[n] = logI[n] - solarmag.mu_to_logI(mu[n]+dmu[n],'F814W_ACS',sz[names[n]])


xx,yy,zz = np.log10(sigsrc), logI, np.log10(re)
dxx,dyy,dzz = dsigsrc/sigsrc/np.log(10.), dlogI.copy(),dre/re/np.log(10.)
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=-0.99*dyy*dzz, -0.99*dyy*dzz

np.save('/data/ljo31b/EELs/esi/kinematics/FP_EELs',np.column_stack((xx,yy,zz,dxx,dyy,dzz)))
# need to get cvariance? Have this between re and mu somewhere already?

pars, cov = [], []
pars.append(pymc.Uniform('a',0.5,2,1.3 ))
pars.append(pymc.Uniform('b',0.1,0.5,0.3 ))
pars.append(pymc.Uniform('alpha',-20.,0.,-10 ))
pars.append(pymc.Uniform('mu x',2,3,2.3 ))
pars.append(pymc.Uniform('mu y',15,30,23 ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,0.5 ))
pars.append(pymc.Uniform('rho',-1.,1,-0.5 ))
pars.append(pymc.Uniform('sigma',0,10,0.01 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.1,0.1,0.01]
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

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=7,nwalkers=28)
S.sample(2000)
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
pl.scatter(zz,a*xx+b*yy+alpha,color='SteelBlue')
Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'a = '+'%.2f'%a)
pl.figtext(0.2,0.75,'b = '+'%.2f'%b)
pl.xlabel(r'$\log R_e$')
pl.ylabel('%.2f'%a+'$\log\sigma$ +'+'%.2f'%b+'$\mu$')
pl.title('the FP of the EELs')
pl.show()

for key in dic.keys():
    pl.figure()
    pl.plot(dic[key])
    pl.title(key)
pl.show()
