import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances

rec,drec,muc,dmuc,sigmac,dsigmac = np.loadtxt('/data/ljo31b/MACSJ0717/data/ComaFP.dat').T
sigmac /=100.
dsigmac /= 100.
logI = solarmag.mu_to_logI(muc,'g_SDSS',0.0231)
dlogI = logI - solarmag.mu_to_logI(muc+dmuc,'g_SDSS',0.0231)

xx,yy,zz = np.log10(sigmac), logI, np.log10(rec)
dxx,dyy,dzz = dsigmac/sigmac/np.log(10.), (dlogI**2. + (2.*drec/rec/np.log(10.))**2.)**0.5,drec/rec/np.log(10.)
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = -0.99*dyy*dzz,-0.99*dyy*dzz
# need to get cvariance? Have this between re and mu somewhere already?
np.save('/data/ljo31b/EELs/esi/kinematics/FP_Coma',np.column_stack((xx,yy,zz,dxx,dyy,dzz)))

pars, cov = [], []
pars.append(pymc.Uniform('a',-2.,2,1.0 ))
pars.append(pymc.Uniform('b',-2.,1,-0.8 ))
pars.append(pymc.Uniform('alpha',-20.,20.,3. ))
pars.append(pymc.Uniform('mu x',-0.5,0.5,0.1 ))
pars.append(pymc.Uniform('mu y',2,4,2.75 ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,0.5 ))
pars.append(pymc.Uniform('rho',-1.,1,-0.5 ))
pars.append(pymc.Uniform('sigma',0,10,0.05 )) 
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


S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=7,nwalkers=28)#nwalkers)#,initialPars = init)
S.sample(2000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_Coma'
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
pl.show()

for key in dic.keys():
    pl.figure()
    pl.plot(dic[key])
    pl.title(key)
pl.show()

