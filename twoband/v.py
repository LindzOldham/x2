import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances

''' data stuff'''
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

# 1 src
phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
re1,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
mu1,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']

# 2 src
phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
re2,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
mu2,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
###################
result = np.load('/data/ljo31b/EELs/FP/inference/zobs_211')
lp,trace,dic,_ = result
for key in dic.keys():
    dic[key] = dic[key][4000:]

alpha,beta,gamma = np.median(dic['a'].ravel()), np.median(dic['b'].ravel()), np.median(dic['alpha'].ravel())

# 1 src
X1 = alpha*np.log10(sigmas) + beta * mu1 + gamma
Y1 = np.log10(re1)
# 2 src
X2 = alpha*np.log10(sigmas) + beta * mu2 + gamma
Y2 = np.log10(re2)

cols = ['SteelBlue','Crimson','DarkOrange','Cyan','SeaGreen','LightPink','Khaki','Purple','k','LightGray','FireBrick','Chartreuse','YellowGreen']

for i in range(X1.size):
    pl.scatter(X1[i],Y1[i],color=cols[i],marker='*',s=80,alpha=0.5)
    pl.scatter(X2[i],Y2[i],color=cols[i],marker='o',s=80,alpha=0.5)

xline=np.linspace(-0.1,1.3,10)
pl.plot(xline,xline)
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%alpha+'$\log\sigma$ +'+'%.2f'%abs(beta)+'$\mu$')
pl.figtext(0.2,0.8,'stars = 211')
pl.figtext(0.2,0.75,'circles = 212')

# now compare huge and non-huge 211 source models
phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
re,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
mu,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
