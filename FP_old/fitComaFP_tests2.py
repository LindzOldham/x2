import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

rec,drec,muc,dmuc,sigmac,dsigmac = np.loadtxt('/data/ljo31b/MACSJ0717/data/ComaFP.dat').T

xx,yy,zz = np.log10(sigmac), muc, np.log10(rec)
dxx,dyy,dzz = dsigmac/sigmac, dmuc.copy(),drec/rec
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# need to get cvariance? Have this between re and mu somewhere already?

pars, cov = [], []
pars.append(pymc.Uniform('a',0.5,2,1.3 ))
pars.append(pymc.Uniform('b',0.1,0.5,0.3 ))
pars.append(pymc.Uniform('alpha',-20.,0.,-10 ))
pars.append(pymc.Uniform('mu x',2,3,2.3 ))
pars.append(pymc.Uniform('mu y',15,30,23 ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,0.5 ))
#pars.append(pymc.Uniform('tau xy',0,10,1 ))
#pars.append(pymc.Uniform('tau yx',0,10,1 )) # assume the same as above?!?
#pars.append(pymc.Uniform('sigma',0,10,0.01 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.1]
optCov = np.array(cov)
sigma=0.

@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    a,b,alpha,mux,muy,taux,tauy = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value
    tauxy,tauyx = 0.,0.
    taux2,tauy2,tauxy2,tauyx2,sigma2,a2,b2 = taux**2.,tauy**2.,tauxy**2.,tauyx**2.,sigma**2.,a**2.,b**2.
    X = zz - alpha - a*mux - b*muy
    Y = xx - mux
    Z = yy - muy
    Sxx = taux2*a2 + a*b*(tauxy2+tauyx2) + tauy2*b2 + sigma2 + szz
    Syy = taux2 + sxx
    Szz = tauy2 + syy
    Sxy = a*taux2 + b*tauyx2 + sxz
    Sxz = a*tauxy2 + b*tauy2 + syz
    Syx = taux2*a + tauxy2*b + sxz
    Syz = tauxy2 + sxy
    Szy = tauyx2 + syx
    Szx = tauxy2*a + tauy2*b + syz
    resid = 0
    args = np.zeros(X.size)
    for ii in range(X.size):
        V = np.matrix([[Sxx[ii], Sxy, Sxz],[Syx,Syy[ii],Syz],[Szx,Szy,Szz[ii]]])
        Vinv = V.I
        Vdet = np.linalg.det(V)
        ZZ = np.matrix([[X[ii],Y[ii],Z[ii]]]).T
        args[ii] = -0.5*np.dot(ZZ.T,np.dot(Vinv,ZZ)) - 0.5*np.log(Vdet)
        resid += args[ii]
    return resid

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=28)
S.sample(4000)
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

a,b,alpha,mux,muy,taux,tauy = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value

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
