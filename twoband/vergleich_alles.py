import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances

magv,magi,magk,v_src,i_src,k_src,muv2,mui,muk = np.load('/data/ljo31/Lens/LensParams/got_phot_212_srcs_z0.npy')
magv,magi,magk,v_src,i_src,k_src,muv1,mui,muk = np.load('/data/ljo31/Lens/LensParams/got_phot_211_srcs_z0.npy')

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
reA,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
muA,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
reB,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
muB,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)

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
sigmas,sigmal,dsigmas,dsigmal = np.delete(sigmas,6), np.delete(sigmal,6),np.delete(dsigmas,6),np.delete(dsigmal,6)

def scat(x,y,col='SteelBlue'):
    pl.scatter(x,y,color=col,s=40,alpha=0.5)

def fp(x,y,z,alpha,beta,fig=True):
    if fig:
        pl.figure()
        scat(alpha*np.log10(x) + beta*y, np.log10(z))
    else:
        scat(alpha*np.log10(x) + beta*y, np.log10(z),col='Crimson')

#
'''fp(sigmas,muv1,reB,1.2,0.3)
pl.title('1 src z0')

fp(sigmas,muv2,reA,1.2,0.3)
pl.title('2 src z0')

fp(sigmas,muB,reB,1.2,0.3)
pl.title('1 src zs')

fp(sigmas,muA,reA,1.2,0.3)
pl.title('2 src zs')


fp(sigmas,muv1,reB,0.4,0.3)
pl.title('1 src z0')

fp(sigmas,muv2,reA,0.4,0.3)
pl.title('2 src z0')

fp(sigmas,muB,reB,0.4,0.3)
pl.title('1 src zs')

fp(sigmas,muA,reA,0.4,0.3)
pl.title('2 src zs')'''

fp(sigmas,muA,reA,1.2,0.3)
fp(sigmas,muB,reB,1.2,0.3,fig=False)

fp(sigmas,muA,reA,0.3,0.3)
fp(sigmas,muB,reB,0.3,0.3,fig=False)
