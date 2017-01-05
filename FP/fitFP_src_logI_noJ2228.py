import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances
from SampleOpt import AMAOpt
from tools import solarmag
from astLib import astCalc
#from EllipsePlot import *

# two ways: one, get L then re in kpc. Or: use logI, but at z=0. Then put L in units of 10**8 solar luminosities!!!

fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results_1.00_lens_bc03_vdfit.npy')
l,m,u = fp
d = np.mean((l,u),axis=0)
dvl,dvs,dsigmal,dsigmas = d.T
vl,vs,sigmal,sigmas = m.T
dsigmas = sigmas*0.05
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

scales = np.array([astCalc.da(sz[name][0])*1e3*np.pi/180./3600. for name in names])
mag = mu - 2.5*np.log10(2.*np.pi*re**2./scales**2.)


sigs = np.load('/data/ljo31/Lens/LensParams/RelogI_covariances.npy')
rho = sigs[:,3]

bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
logI,dlogI = mu*0,mu*0
logL = mu*0
logLR = mu*0.

for ii in range(len(names)):
    band = bands[names[ii]]
    #logI[ii] = np.log10(solarmag.mu_to_I(mu[ii],band+'_ACS',0))#sz[names[ii]][0]))
    #dlogI[ii] = logI[ii] - solarmag.mu_to_logI(mu[ii]+dmu[ii],band+'_ACS',0)#sz[names[ii]][0])
    logL[ii] = solarmag.mag_to_logL(mag[ii],band+'_ACS',sz[names[ii]][0]) - 8.
    logLR[ii] =  np.log10(2.*np.pi*re[ii]**2.)
    logI[ii] =  logL[ii] - logLR[ii]

dlogI = 0.4*dmu

#xx,yy,zz = np.log10(sigmas), logI.copy(), np.log10(re)
xx,yy,zz = np.log10(sigmas), logI.copy(), np.log10(re)
xo,yo,zo = xx.copy(),yy.copy(),zz.copy()
dxx,dyy,dzz = dsigmas/sigmas/np.log(10.), dlogI.copy(), dre/re/np.log(10.)
dxo,dyo,dzo = dxx.copy(),dyy.copy(),dzz.copy()
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
xx,yy,zz,sxx,syy,szz=xx[:-1],yy[:-1],zz[:-1],sxx[:-1],syy[:-1],szz[:-1]
#xx,yy,zz,sxx,syy,szz = np.delete(xx,3),np.delete(yy,3),np.delete(zz,3),np.delete(sxx,3),np.delete(syy,3),np.delete(szz,3)
#xx,yy,zz,sxx,syy,szz = np.delete(xx,8),np.delete(yy,8),np.delete(zz,8),np.delete(sxx,8),np.delete(syy,8),np.delete(szz,8)

# covariances
syz,szy = rho*dyy*dzz,rho*dyy*dzz
#np.save('/data/ljo31b/EELs/esi/kinematics/FP_EELs_mu',np.column_stack((xx,yy,zz,dxx,dyy,dzz)))

'''
xx,yy,zz,dxx,dyy,dzz = np.load('/data/ljo31b/EELs/esi/kinematics/FP_EELs.npy').T
sigs = np.load('/data/ljo31/Lens/LensParams/RelogI_covariances.npy')
rho = sigs[:,3]
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = rho*dyy*dzz,rho*dyy*dzz

'''

pars, cov = [], []
pars.append(pymc.Uniform('a',-2.,3,1 ))
pars.append(pymc.Uniform('b',-5.,5 ))
pars.append(pymc.Uniform('alpha',-20.,20.))
pars.append(pymc.Uniform('mu x',-0.5,1.0,0.3 ))
pars.append(pymc.Uniform('mu y',-10.,30. ))
pars.append(pymc.Uniform('tau x',0,1,0.1 ))
pars.append(pymc.Uniform('tau y',0,6,0.3 ))
pars.append(pymc.Uniform('rho',-1.,1 ))
pars.append(pymc.Uniform('sigma',0,2,0.1 )) 
cov += [1.,1.,1.,0.5,0.5,0.05,0.1,0.1,0.05]
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
S = myEmcee.Emcee(pars+[likelihood],cov=optCov/5.,nthreads=4,nwalkers=28)
S.sample(1300)
outFile = '/data/ljo31b/EELs/FP/inference/FP_src_logI_noJ2228'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[-300:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

a,b,alpha,mux,muy,taux,tauy,rho,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[5].value, pars[6].value,pars[7].value,pars[8].value
da, db = a - np.percentile(dic['a'][-300:].ravel(),16), b -np.percentile(dic['b'][-300:].ravel(),16)

pl.figure()
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')

Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline)
pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%a+'$\log\sigma$ '+'%.2f'%b+'$\log I$')

pl.title('EELs')
pl.scatter(a*xo[-1]+b*yo[-1]+alpha,zo[-1],color='Crimson')
pl.scatter(a*xo[3]+b*yo[3]+alpha,zo[3],color='Crimson')

#pl.savefig('/data/ljo31b/EELs/esi/TeX/eelsFPmu.pdf')
pl.show()

sig_j2228 = (a*xo[-1]+b*yo[-1]+alpha-zo[-1])/(dzo[-1]**2. + sigma**2.)**0.5
print sig_j2228
sig_j1125 = (a*xo[3]+b*yo[3]+alpha-zo[3])/(dzo[3]**2. + sigma**2.)**0.5
print sig_j1125
sigs = (a*xo+b*yo+alpha-zo)/(dzo**2. + sigma**2.)**0.5
print sigs

#### plot the FP with error ellipses and instrinsic scatter
# first pass: errorbars and intrinsic scatter
dxx,dyy,dzz = dxx[:-1],dyy[:-1],dzz[:-1]
gamma16,gamma84 = np.percentile(dic['alpha'][-300:].ravel(),16),np.percentile(dic['alpha'][-300:].ravel(),84)
dgamma = np.mean((alpha-gamma16,gamma84-alpha))
print dgamma, sigma

# for each step in chain get uncertainty!
#burnin = 500
#f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
#fits=np.zeros((len(f),xfit.size))
#for j in range(0,len(f)):
#    a,b,alpha,mux,muy,taux,tauy,rho,sigma = f[j]
 #   fits[j] = a*xfit+b*
# this is complicated because of the dimensionality...


pl.figure()
pl.errorbar(a*xx+b*yy+alpha,zz,xerr=((a*dxx)**2. + (b*dyy)**2.)**0.5, yerr=dzz,color='SteelBlue',fmt='o')
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')
Zline = np.linspace(min(zz)-1,max(zz)+1,10)
pl.plot(Zline,Zline,color='SteelBlue',label='observed')
pl.fill_between(Zline,Zline,Zline+sigma,color='LightBlue',alpha=0.5)
pl.fill_between(Zline,Zline,Zline-sigma,color='LightBlue',alpha=0.5)
#pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
#pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'log R$_e$')
pl.xlabel('%.2f'%a+'log$\sigma$ '+'%.2f'%b+'log I')
pl.axis([0.06,1.0,0,1.05])

'''diff = (sigma**2. + dgamma**2.)**0.5
pl.figure()
pl.errorbar(a*xx+b*yy+alpha,zz,xerr=((a*dxx)**2. + (b*dyy)**2.)**0.5, yerr=dzz,color='SteelBlue',fmt='o')
pl.scatter(a*xx+b*yy+alpha,zz,color='SteelBlue')
Zline = np.linspace(min(zz),max(zz),10)
pl.plot(Zline,Zline,color='SteelBlue',label='observed')
pl.fill_between(Zline,Zline,Zline+diff,color='LightBlue',alpha=0.5)
pl.fill_between(Zline,Zline,Zline-diff,color='LightBlue',alpha=0.5)
pl.figtext(0.2,0.8,'$a = '+'%.2f'%a+'\pm'+'%.2f'%da+'$')
pl.figtext(0.2,0.75,'$b = '+'%.2f'%b+'\pm'+'%.2f'%db+'$')
pl.ylabel(r'log R$_e$')
pl.xlabel('%.2f'%a+'log$\sigma$ '+'%.2f'%b+'$log I')

#pl.scatter(a*xo[-1]+b*yo[-1]+alpha,zo[-1],color='Crimson')

#pl.savefig('/data/ljo31b/EELs/esi/TeX/eelsFPmu.pdf')'''
pl.show()
