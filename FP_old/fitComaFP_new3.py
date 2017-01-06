import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

alp = np.random.randn(100)*0.5 + 1.25
bet = np.random.randn(100)*0.5 + 0.6
gam = np.random.randn(100)*3.-7.

logsigma = np.random.randn(300)*0.5 + 2.35
mu = np.random.randn(300)*1. + 20.
params = []

for II in range(48):
    logre = alp[II]*logsigma + bet[II]*mu + gam[II] + np.random.randn(300)*0.2

    xx,yy,zz = logsigma, mu,logre
    dxx,dyy,dzz = xx*0.05,yy*0.05,zz*0.05
    sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
    syz,szy = 0.,0.
    sxy,syx,sxz,szx = 0,0,0,0
    syz,szy=0,0

    pars, cov = [], []
    pars.append(pymc.Uniform('a',-1.,5,1.25 ))
    pars.append(pymc.Uniform('b',-1.,3.,0.3 ))
    pars.append(pymc.Uniform('alpha',-20.,10.,-7. ))
    pars.append(pymc.Uniform('mu x',2,3,2.3 ))
    pars.append(pymc.Uniform('mu y',15,30,20 ))
    pars.append(pymc.Uniform('tau x',0,1,0.5 ))
    pars.append(pymc.Uniform('tau y',0,6,1. ))
    #pars.append(pymc.Uniform('tau xy',0,10,1 ))
    #pars.append(pymc.Uniform('tau yx',0,10,1 )) # assume the same as above?!?
    pars.append(pymc.Uniform('sigma',0,1,0.005 )) 
    cov += [1.,1.,1.,0.5,0.5,0.05,0.1,0.005]
    optCov = np.array(cov)

    @pymc.deterministic
    def logP(value=0.,p=pars):
        lp=0.
        a,b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value
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
        for ii in range(X.size):
            V = np.array([[Sxx[ii], Sxy, Sxz],[Syx,Syy[ii],Syz],[Szx,Szy,Szz[ii]]])
            Vinv = np.linalg.inv(V)
            Vdet = np.linalg.det(V)
            ZZ = np.array([[X[ii],Y[ii],Z[ii]]]).T
            args = -0.5*np.dot(ZZ.T,np.dot(Vinv,ZZ)) - 0.5*np.log(Vdet)
            resid += args
        return resid

    @pymc.observed
    def likelihood(value=0.,lp=logP):
        return lp

    '''S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=10,nwalkers=28)
    S.sample(1000)
    outFile = '/data/ljo31b/EELs/FP/inference/FP_Comatest'+str(II)
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()
    result = S.result()'''
    result = np.load('/data/ljo31b/EELs/FP/inference/FP_Comatest'+str(II))

    lp,trace,dic,_ = result
    a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
    print trace.shape
    trace = trace[500:]
    ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
    for i in range(len(pars)):
        pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
    
    a,b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value
    print '%.2f'%a,'%.2f'%b,'%.2f'%alpha
    print '%.2f'%alp[II],'%.2f'%bet[II],'%.2f'%gam[II]

    params.append(np.array([alp[II],bet[II],gam[II],a,b,alpha,mux,muy,taux,tauy,sigma]))
    np.save('/data/ljo31b/EELs/FP/inference/FP_params_3',np.array(params))

params = np.array(params)
np.save('/data/ljo31b/EELs/FP/inference/FP_params_3',params)

pl.figure()
pl.scatter(params[:,0],params[:,3])
pl.title('alpha')

pl.figure()
pl.scatter(params[:,0],params[:,4])
pl.title('beta')

pl.figure()
pl.scatter(params[:,0],params[:,5])
pl.title('gamma')
pl.show()

params = np.load('/data/ljo31b/EELs/FP/inference/FP_params_3.npy')
params2 = np.load('/data/ljo31b/EELs/FP/inference/FP_params_2.npy')

pl.figure()
pl.scatter(params2[:48,0],params[:48,3],color='SteelBlue',s=40)
xline=np.linspace(0,2.5,10)
pl.plot(xline,xline,color='SteelBlue')
pl.xlabel(r'$\alpha_{true}$')
pl.ylabel(r'$\alpha_{model}$')

pl.figure()
pl.scatter(params2[:48,1],params[:48,4],color='SteelBlue',s=40)
xline=np.linspace(-1,2.5,10)
pl.plot(xline,xline,color='SteelBlue')
pl.xlabel(r'$\beta_{true}$')
pl.ylabel(r'$\beta_{model}$')

pl.figure()
pl.scatter(params2[:48,2],params[:48,5],color='SteelBlue',s=40)
xline=np.linspace(-16,2,10)
pl.plot(xline,xline,color='SteelBlue')
pl.xlabel(r'$\gamma_{true}$')
pl.ylabel(r'$\gamma_{model}$')
