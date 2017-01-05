import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances
from SampleOpt import AMAOpt
from astLib import astCalc

# what happens if we say there's no DM in these systems? -- alpha = 1.4
# the fact that even with no DM we don't get the virial plane means there must be further stellar population and structural trends?

fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results_0.30_source_indous_vdfit_jul2016_J2228.npy')
l,m,u = fp
d = np.mean((l,u),axis=0)
dvl,dvs,dsigmal,dsigmas = d.T
vl,vs,sigmal,sigmas = m.T
dsigmas = sigmas*0.05
dsigmas[-2:] = sigmas[-2:]*0.1
sigmas /= 100.
sigmal /= 100.
dsigmas /= 100.
dsigmal /= 100.
# remove J1248 as we don't have photometry
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_lens_vdfit.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_1.00_lens_vdfit.npy')[()]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_huge_new_new.fits')[1].data
names = phot['name']

re,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
mu,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)

sigs = np.load('/data/ljo31/Lens/LensParams/ReMu_covariances.npy')
rho = sigs[:,3]

logI = np.load('/data/ljo31/Lens/LensParams/V_redshift0_model.npy')

logIold = np.load('/data/ljo31/Lens/LensParams/logI_obsframe.npy')

logI -= 8.5
#logI = logIold
dlogI = 0.1*np.ones(logI.size)
#print logI

VV,VVup,VVlo,V,Vup,Vlo = np.load('/data/ljo31/Lens/LensParams/F606W_rightredshifts_model_marginalised.npy').T
logI = VV-8.5
dlogI = np.mean((VVup-VV,VV-VVlo),0)

#for ii in range(len(names)):
#    print names[ii], '%.2f'%logI[ii], '%.2f'%dlogI[ii]

#for ii in range(len(names)):
#    print names[ii], '%.2f'%re[ii], '%.2f'%dre[ii], '%.2f'%(sigmas[ii]*100), '%.2f'%(dsigmas[ii]*100)


xx,yy,zz = np.log10(sigmas), logI, np.log10(re)
xo,yo,zo = xx.copy(),yy.copy(),zz.copy()
dxx,dyy,dzz = dsigmas/sigmas/np.log(10.), dlogI, dre/re/np.log(10.)
dxo,dyo,dzo = dxx.copy(),dyy.copy(),dzz.copy()
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
syz,szy = rho*dyy*dzz,rho*dyy*dzz
syz,szy = np.zeros(dyy.size), np.zeros(dyy.size)#rho*dyy*dzz,rho*dyy*dzz

#xx,yy,zz,sxx,syy,szz=xx[6:-1],yy[6:-1],zz[6:-1],sxx[6:-1],syy[6:-1],szz[6:-1]
#syz,szy = rho[6:-1]*dyy[6:-1]*dzz[6:-1],rho[6:-1]*dyy[6:-1]*dzz[6:-1]
#xx,yy,zz,sxx,syy,szz=xx[:7],yy[:7],zz[:7],sxx[:7],syy[:7],szz[:7]
#syz,szy = rho[:7]*dyy[:7]*dzz[:7],rho[:7]*dyy[:7]*dzz[:7]
#print len(xx)
#print zz




#I=2
#names = names[6:-1]
#print names[I]

I=-3 # J1606
names = np.delete(names,I)
yy,zz,syy,szz = np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
#I=-2 # J1619
#yy,zz,syy,szz = np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
#names = np.delete(names,I)
#I = -1 # J2228
#xx,sxx,yy,zz,syy,szz = np.delete(xx,I),np.delete(sxx,I),np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
#names = np.delete(names,I)
#I = -1 # J1619
#xx,sxx,yy,zz,syy,szz = np.delete(xx,I),np.delete(sxx,I),np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
#names = np.delete(names,I)
#I = 7 # J0901
#xx,sxx,yy,zz,syy,szz = np.delete(xx,I),np.delete(sxx,I),np.delete(yy,I),np.delete(zz,I),np.delete(syy,I),np.delete(szz,I)
#names = np.delete(names,I)


pars, cov = [], []
pars.append(pymc.Uniform('a',-10.,10,1.0 ))
#a=1.5
pars.append(pymc.Uniform('b',-5.,5,-0.8 ))
pars.append(pymc.Uniform('alpha',-20.,20.,-6. ))
pars.append(pymc.Uniform('mu x',-2,2.0,0.3 ))
pars.append(pymc.Uniform('mu y',-10.,30. ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,3,1.0 ))
pars.append(pymc.Uniform('rho',-1.,1 ))
pars.append(pymc.Uniform('sigma',0,10,0.2 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.1,0.1,0.1]
#cov += [1.,1.,0.5,0.5,0.05,0.1,0.1,0.1]

optCov = np.array(cov)


@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    a,b,alpha,mux,muy,taux,tauy,rho,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value, pars[8].value
    #b,alpha,mux,muy,taux,tauy,rho,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value
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

# optimise first!

SS = AMAOpt(pars,[likelihood],[logP],cov=cov)
SS.sample(4000)
lp,trace,det = SS.result()
#pl.figure()
#pl.plot(lp)
#pl.show() 
print 'results from optimisation:'
for i in range(len(pars)):
    pars[i].value = trace[-1,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

print 'new!'
S = myEmcee.Emcee(pars+[likelihood],cov=optCov/5.,nthreads=24,nwalkers=28)
S.sample(5000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_logI_CHECK'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[2000:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

a,b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value
#b,alpha,mux,muy,taux,tauy,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value
#da, db = np.percentile(dic['a'][2000:].ravel(),84)-a, np.percentile(dic['b'][2000:].ravel(),84)-b
dalpha = np.percentile(dic['alpha'][2000:].ravel(),84)-alpha

yerr = szz**0.5
xerr = (a**2.*sxx + b**2*syy)**0.5
pl.figure()

Zline = np.linspace(min(zz)-0.5,max(zz)+0.5,10)
pl.plot(Zline,Zline,color='k')
pl.fill_between(Zline,Zline,Zline+sigma,color='LightGray')
pl.fill_between(Zline,Zline,Zline-sigma,color='LightGray')

pl.scatter(a*xx+b*yy+alpha,zz,color='k')
pl.errorbar(a*xx+b*yy+alpha,zz,xerr=xerr,yerr=yerr,color='k',fmt='o')
#pl.scatter(a*xo[-1]+b*yo[-1]+alpha,zo[-1],color='r',s=50)
#pl.scatter(a*xo[-2]+b*yo[-2]+alpha,zo[-2],color='r',s=50)

#pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
#pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'log r$_e$')
pl.xlabel('%.2f'%a+'log$\sigma$ '+'-'+'%.2f'%abs(b)+'log I$_e$'+'+'+'%.2f'%alpha)

#pl.savefig('/data/ljo31b/EELs/esi/TeX/eelsFPmu.pdf')
#pl.axis([0,1.1,0,1.1])

# also a face-on view of the plane!
# plot the face-on projection
xerr = (a**2 * sxx + b**2 * syy + (a**2+b**2)*szz)**0.5
yerr = (b**2*sxx + a**2 * syy)**0.5


pl.figure()
pl.scatter(a*xx + b*yy + (a**2 + b**2)**0.5 * zz, -b*xx + a*yy, color='k')
pl.errorbar(a*xx + b*yy + (a**2 + b**2)**0.5 * zz, -b*xx + a*yy, xerr=xerr, yerr=yerr,color='k',fmt='o')
pl.xlabel('%.2f'%a+r'log$\sigma$ '+'%.2f'%b+r'log I$_e$ + '+'%.2f'%(np.sqrt(a**2.+b**2.))+r'log r$_e$')
pl.ylabel('%.2f'%(-1*b)+r'log$\sigma$ + '+'%.2f'%(a)+r'log I$_e$')
#pl.axis([-0.6,2.0,0,2.2])

# a second side-on projection
xerr = (szz + b**2 * syy)**0.5
yerr = (sxx)**0.5

pl.figure()
pl.plot(a*Zline+alpha,Zline,color='k')
pl.fill_between(a*Zline+alpha,Zline+0.3*dalpha,Zline-0.3*dalpha,color='LightGray')
print dalpha
pl.scatter(zz - b*yy, xx,color='k')
pl.errorbar(zz-b*yy,xx,xerr=xerr,yerr=yerr,color='k',fmt='o')

pl.xlabel('log r$_e$ +'+'%.2f'%abs(b)+'log I$_e$')
pl.ylabel('log $\sigma$')
#pl.axis([0.8,1.35,0,0.6])
pl.show()
