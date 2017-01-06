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
sigmas,sigmal,dsigmas,dsigmal = np.delete(sigmas,(6,7,9,12)), np.delete(sigmal,(6,7,9,12)),np.delete(dsigmas,(6,7,9,12)),np.delete(dsigmal,(6,7,9,12))
# second deletion is J1323 as this IS A SPIRAL GALAXY!
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
NAMES = lz.keys()
NAMES.sort()
NAMES = np.delete(NAMES,(6,7,9,12))

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_lensgals_huge.fits')[1].data
names = phot['name']
phot2 = py.open('/data/ljo31/Lens/LensParams/Phot_1src_lensgals_huge_extras.fits')[1].data
names2 = phot2['name']

PHOT = []
for name in NAMES:
    ii=np.where(names==name)
    if len(names[ii])>0:
        PHOT.append(phot[ii])
    else:
        jj = np.where(names2==name)
        PHOT.append(phot2[jj])
    print ii, name

re = [PHOT[i]['Re v'][0] for i in range(len(NAMES))]
mu = [PHOT[i]['mu v'][0] for i in range(len(NAMES))]
mul = [PHOT[i]['mu v lo'][0] for i in range(len(NAMES))]
muu = [PHOT[i]['mu v hi'][0] for i in range(len(NAMES))]
rel = [PHOT[i]['Re v lo'][0] for i in range(len(NAMES))]
reu = [PHOT[i]['Re v hi'][0] for i in range(len(NAMES))]
dre = np.mean((rel,reu),axis=0)
dmu = np.mean((mul,muu),axis=0)
re,mu=np.array(re),np.array(mu)

rho = np.load('/data/ljo31/Lens/LensParams/rho_huge_211_lensgals.npy')
rho = np.delete(rho,(6,8,11))
N=13
print NAMES[:N]
xx,yy,zz = np.log10(sigmal)[:N], mu.copy()[:N], np.log10(re)[:N]
dxx,dyy,dzz = dsigmal[:N]/sigmal[:N]/np.log(10.), dmu[:N], dre[:N]/re[:N]/np.log(10.)
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = rho[:N]*dyy*dzz,rho[:N]*dyy*dzz
#np.save('/data/ljo31b/EELs/esi/kinematics/FP_EELs_lenses',np.column_stack((xx,yy,zz,dxx,dyy,dzz)))
# need to get cvariance? Have this between re and mu somewhere already?


pars, cov = [], []
pars.append(pymc.Uniform('a',-2.,3,1.0 ))
pars.append(pymc.Uniform('b',-1.,1,0.3 ))
pars.append(pymc.Uniform('alpha',-20.,20.,-6. ))
pars.append(pymc.Uniform('mu x',-0.5,1.0,0.3 ))
pars.append(pymc.Uniform('mu y',1.,30. ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,1.0 ))
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


S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=8,nwalkers=28)
S.sample(6000)
outFile = '/data/ljo31b/EELs/FP/inference/XX'#FP_lens_mu_211_huge'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
#result = np.load('/data/ljo31b/EELs/FP/inference/FP_lens_mu_212')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[4000:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

for key in dic.keys():
    dic[key] = dic[key][4000:]


a,b,alpha,mux,muy,taux,tauy = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value
da, db = a - np.percentile(dic['a'].ravel(),16), b -np.percentile(dic['b'].ravel(),16)

pl.figure()
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')
Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ +'+'%.2f'%b+'$\mu$')

pl.title('EELs lenses')
pl.savefig('/data/ljo31b/EELs/esi/TeX/lensFPmu.pdf')
pl.show()
