import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

rec,drec,muc,dmuc,sigmac,dsigmac = np.loadtxt('/data/ljo31b/MACSJ0717/data/ComaFP.dat').T

z,x,y = np.log10(rec),np.log10(sigmac),muc.copy()
dz,dx,dy = drec/rec/np.log(10.), dsigmac/sigmac/np.log(10.), dmuc.copy()

pars,cov = [],[]
pars.append(pymc.Uniform('alpha',0.0,2.0,1.3 ))
pars.append(pymc.Uniform('beta',0.1,2.0,0.3 ))
pars.append(pymc.Uniform('gamma',-20.,10.,-10 ))
cov+= [0.1,0.1,0.1]
cov = np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    alpha,beta,gamma = pars[0].value, pars[1].value, pars[2].value
    Delta = (z - alpha*x - beta*y - gamma)**2./(dx**2. + dy**2. + dz**2.)#/(alpha**2. + beta**2. + 1.)/(dx**2. + dy**2. + dz**2.)
    lp = -1.*np.sum(Delta)
    return lp


@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=cov,nthreads=7,nwalkers=28)
S.sample(1000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_Coma_jorgensen96'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
result = np.load('/data/ljo31b/EELs/FP/inference/FP_Coma_jorgensen96')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

alpha,beta,gamma = pars[0].value, pars[1].value, pars[2].value

pl.figure()
pl.scatter(alpha*x+beta*y+gamma,z,color='SteelBlue')
Zline = np.linspace(min(z),max(z),10)
pl.plot(Zline,Zline)
#pl.figtext(0.2,0.8,'$\alpha$ = '+'%.2f'%alpha)
#pl.figtext(0.2,0.75,'$\beta$ = '+'%.2f'%beta)
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%alpha+'$\log\sigma$ +'+'%.2f'%beta+'$\mu$')
pl.show()
'''
for key in dic.keys():
    pl.figure()
    pl.plot(dic[key])
    pl.title(key)
pl.show()
'''
