import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances

def plot_an_fp(ap,col):
    fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results_'+str(ap)+'.npy')
    l,m,u = fp
    d = np.mean((l,u),axis=0)
    dvl,dvs,dsigmal,dsigmas = d.T
    vl,vs,sigmal,sigmas = m.T
    sigmas /= 100.
    sigmal /= 100.
    dsigmas /= 100.
    dsigmal /= 100.
    # remove J1248 as we don't have photometry
    sigmas,sigmal,dsigmas,dsigmal = np.delete(sigmas,0), np.delete(sigmal,0),np.delete(dsigmas,0),np.delete(dsigmal,0)
    sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_'+str(ap)+'.npy')[()]
    NAMES = sz.keys()
    NAMES.sort()

    phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_huge_new.fits')[1].data
    idxs = [6,7,8,10,11,12]
    
    re,rel,reu = phot['Re v'][idxs], phot['Re v lo'][idxs], phot['Re v hi'][idxs]
    dre = np.mean((rel,reu),axis=0)
    mu,mul,muu = phot['mu v'][idxs], phot['mu v lo'][idxs], phot['mu v hi'][idxs]
    dmu = np.mean((mul,muu),axis=0)

    xx,yy,zz = np.log10(sigmas), mu.copy(), np.log10(re)
    dxx,dyy,dzz = dsigmas/sigmas/np.log(10.), dmu, dre/re/np.log(10.)
    sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
    syz,szy = 0.,0.
    sxy,syx,sxz,szx = 0,0,0,0
    syz,szy = 0.9*dyy*dzz,0.9*dyy*dzz

    a = 0.7
    b = 0.3
    pl.scatter(a*xx+b*yy,zz,c=col,s=50)
    pl.ylabel(r'$\log R_e$')
    pl.xlabel('%.2f'%a+'$\log\sigma$ +'+'%.2f'%abs(b)+'$\mu$')
    pl.title('EELs')



pl.figure()
aps = [0.75,1.0,1.5,1.75,2.0]
colours = ['SteelBlue','Crimson','Navy','DarkOrange','SeaGreen']
for i in range(len(aps)):
    plot_an_fp(aps[i],colours[i])
pl.title('changing the aperture has a tiny effect on the FP')
