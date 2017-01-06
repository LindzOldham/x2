import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

rec,drec,muc,dmuc,sigmac,dsigmac = np.loadtxt('/data/ljo31b/MACSJ0717/data/ComaFP.dat').T

y,X,Y = np.log10(rec), np.log10(sigmac), muc
sigy,sigXX,sigYY = drec/rec/np.log(10.), dsigmac/sigmac/np.log(10.), dmuc
sigXY,sigYX = sigy*0.,sigy*0.
sigyX,sigyY = sigy*0.,sigy*0.

pars, cov = [], []
pars.append(pymc.Uniform('a',0.5,2,1.3 ))
pars.append(pymc.Uniform('b',0.1,0.5,0.3 ))
pars.append(pymc.Uniform('alpha',-20.,0.,-10 ))
pars.append(pymc.Uniform('sigma',0,10,0.01 )) 
pars.append(pymc.Uniform('mu x',2,3,2.3 ))
pars.append(pymc.Uniform('mu y',15,30,23 ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,0.5 ))
#pars.append(pymc.Uniform('tau xy',0,10,1 ))
#pars.append(pymc.Uniform('tau yx',0,10,1 )) # assume the same as above?!?
cov += [1.,1.,1.,0.01,0.5,0.5,0.05,0.1]
optCov = np.array(cov)

tauXY,tauYX=0.,0.
sigXY,sigYX=0.,0.

ndim = 3.
norm = (2*np.pi)**ndim

@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    a,b,alpha,sigma,muX,muY,tauXX,tauYY = pars[0].value,pars[1].value,pars[2].value,pars[3].value,pars[4].value,pars[5].value,pars[6].value,pars[7].value
    
    mu = np.matrix([[muX],[muY]])
    beta = np.matrix([[a],[b]])
    T = np.matrix([[tauXX**2.,tauXY**2.],[tauYX**2.,tauYY**2.]])
    eta = np.matrix([[alpha + np.dot(beta.T,mu).item()],[mu[0].item()],[mu[1].item()]])

    # speed this up a lot by doing it all at once in a list of matrices!
    for ii in range(X.size):
        Sigma = np.matrix([[sigXX[ii]**2.,sigXY**2.],[sigYX**2.,sigYY[ii]**2.]])
        var2d = np.asarray(T + Sigma)
        sigxy = np.matrix([[sigyX[ii]],[sigyY[ii]]])
        LH = np.asarray(np.dot(T,beta) + sigxy)
        RH = np.asarray(np.dot(beta.T,T) + sigxy.T)
        top = np.dot(beta.T,np.dot(T,beta)) + sigma**2. + sigy[ii]**2.

        var=[[top.item(),RH[0][0],RH[0][1]], [LH[0][0],var2d[0][0],var2d[0][1]], [LH[1][0],var2d[1][0],var2d[1][1]]]
        var = np.matrix(var)
        det,ivar = np.linalg.det(var), np.linalg.inv(var)
        
        z = np.matrix([[y[ii]],[X[ii]],[Y[ii]]])

        lp += -0.5*np.dot((z-eta).T, np.dot(ivar,(z-eta))) - 0.5*np.log(det*norm)
    return lp


@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

'''S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=7,nwalkers=28)
S.sample(3000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_Coma_sigmazero'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()'''
result = np.load('/data/ljo31b/EELs/FP/inference/FP_Coma_sigmazero')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

a,b,alpha,sigma,mux,muy,taux,tauy = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value

pl.figure()
pl.scatter(a*X+b*Y+alpha,y,color='SteelBlue')
Zline = np.linspace(min(y),max(y),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'a = '+'%.2f'%a)
pl.figtext(0.2,0.75,'b = '+'%.2f'%b)
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ +'+'%.2f'%b+'$\mu$')
pl.title('the FP of the Coma')
pl.show()
'''
for key in dic.keys():
    pl.figure()
    pl.plot(dic[key])
    pl.title(key)
pl.show()
'''
