import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

re_norm, vd_norm, mu_norm = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_normals_100.dat').T
re_nugg, vd_nugg, mu_nugg = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_nuggets_100.dat').T

# scatter by Gaussian errors with sigma = 1% of observable
f_norm = 10**(-0.4*mu_norm)
f_nugg = 10**(-0.4*mu_nugg)
df_norm = np.random.randn(re_norm.size)*0.03*f_norm
df_nugg = np.random.randn(re_nugg.size)*0.03*f_nugg
f_norm += np.random.randn(re_norm.size)*0.03*f_norm
f_nugg += np.random.randn(re_nugg.size)*0.03*f_nugg
mu_norm = -2.5*np.log10(f_norm)
mu_nugg = -2.5*np.log10(f_nugg)
dmu_norm = df_norm/f_norm
dmu_nugg = df_nugg/f_nugg

re_norm += np.random.randn(re_norm.size)*0.03*re_norm
vd_norm += np.random.randn(re_norm.size)*0.03*vd_norm

re_nugg += np.random.randn(re_norm.size)*0.03*re_nugg
vd_nugg += np.random.randn(re_norm.size)*0.03*vd_nugg

dre_norm = np.random.randn(re_norm.size)*0.03*re_norm
dvd_norm = np.random.randn(re_norm.size)*0.03*vd_norm

dre_nugg = np.random.randn(re_norm.size)*0.03*re_nugg
dvd_nugg = np.random.randn(re_norm.size)*0.03*vd_nugg

pl.figure()
pl.scatter(1.2*np.log10(vd_norm)+0.3*mu_norm, np.log10(re_norm),color='SteelBlue',s=40,label='normals')
pl.scatter(1.2*np.log10(vd_nugg)+0.3*mu_nugg, np.log10(re_nugg),color='Crimson',s=40,label='nuggets')
pl.xlabel('$1.2 \log\sigma + 0.3 \mu$')
pl.ylabel('$\log r_e$')
pl.legend(loc='lower right')

pl.figure()
pl.scatter(0.7*np.log10(vd_norm)+0.3*mu_norm, np.log10(re_norm),color='SteelBlue',s=40,label='normals')
pl.scatter(0.7*np.log10(vd_nugg)+0.3*mu_nugg, np.log10(re_nugg),color='Crimson',s=40,label='nuggets')
pl.xlabel('$0.7 \log\sigma + 0.3 \mu$')
pl.ylabel('$\log r_e$')
pl.legend(loc='lower right')
pl.show()
# nuggets generally have much brighter SBs (as more compact), then small sizes and fewer small VDs.
'''
# now fit the FP for each population in turn. First, the normals
xx,yy,zz = np.log10(vd_norm), mu_norm.copy(), np.log10(re_norm)
dxx,dyy,dzz = dvd_norm/vd_norm/np.log(10.), dmu_norm,dre_norm/re_norm/np.log(10.)
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = 0.85*dyy*dzz,0.85*dyy*dzz

pars, cov = [], []
pars.append(pymc.Uniform('a',-2.,3,1.0 ))
pars.append(pymc.Uniform('b',-1.,1,0.3 ))
pars.append(pymc.Uniform('alpha',-40.,40. ))
pars.append(pymc.Uniform('mu x',1.,3. ))
pars.append(pymc.Uniform('mu y',-30.,-15. ))
pars.append(pymc.Uniform('tau x',0,1 ))
pars.append(pymc.Uniform('tau y',0,6 ))
pars.append(pymc.Uniform('rho',-1.,1,-0.5 ))
pars.append(pymc.Uniform('sigma',0,1,0.2 )) 
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


S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=8,nwalkers=28)
S.sample(6000)
outFile = '/data/ljo31b/EELs/inference/FP/jeansmodels'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
#result = np.load('/data/ljo31b/EELs/FP/inference/zhalf_211')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[5000:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

a,b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value
for key in dic.keys():
    dic[key] = dic[key][4000:]

da, db = a - np.percentile(dic['a'].ravel(),16), b -np.percentile(dic['b'].ravel(),16)

pl.figure()
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')
Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ +'+'%.2f'%abs(b)+'$\mu$')
pl.title('EELs')
pl.show()
'''
