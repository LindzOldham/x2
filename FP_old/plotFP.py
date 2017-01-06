import numpy as np, pylab as pl, pyfits as py

names = ['J0837','J0901','J0913','J1125','J1144','J1218']
vdl,vd,vdu = np.load('/data/ljo31b/EELs/esi/kinematics/inference/night1.npy')
dvd = np.mean((vdl,vdu),axis=0)
vlens,siglens,vsrc,sigsrc = vd.T
dvlens,dsiglens,dvsrc,dsigsrc = dvd.T

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
allnames = phot['name']

re,rel,reu = phot['Re i'], phot['Re i lo'], phot['Re i hi']
dre = np.mean((rel,reu),axis=0)

mu,mul,muu = phot['mu i'], phot['mu i lo'], phot['mu i hi']
dmu = np.mean((mul,muu),axis=0)

re,dre,mu,dmu = re[:6],dre[:6],mu[:6],dmu[:6]
mu -= 10.*np.log10(1.55)

pl.figure()
pl.scatter(np.log10(re),1.2*np.log10(sigsrc)+0.3*mu,color='SteelBlue',s=40)

# compare with Coma and MACSJ0717 -- are they reasonable?

### Coma
rec,drec,muc,dmuc,sigmac,dsigmac = np.loadtxt('/data/ljo31b/MACSJ0717/data/ComaFP.dat').T
# select things with size errors < 5%
#ii=np.where(drec/rec<0.1)
#rec,drec,muc,dmuc,sigmac,dsigmac = rec[ii],drec[ii],muc[ii],dmuc[ii],sigmac[ii],dsigmac[ii]

pl.scatter(np.log10(rec),1.2*np.log10(sigmac)+0.3*muc,color='Crimson',s=40)
