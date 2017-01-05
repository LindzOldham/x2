import numpy as np, pylab as pl, pyfits as py, cPickle
from tools.EllipsePlot import *


# from findML_SEDs - inferred ages for source galaxies from photometry and used to calculate masses etc
logRe,logM,dlogR,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_2src_new.npy').T
logRe1,logM1,dlogR1,dlogM1,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_1src_new.npy').T
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']

col = ['Navy','Yellow','Crimson','SteelBlue','CornflowerBlue','SeaGreen','Cyan','LightPink','Gray','BlueViolet','Chartreuse','DarkSalmon','DarkGoldenRod']
pl.figure()
for i in range(name.size):
    pl.errorbar(logM[i],logRe[i],xerr=dlogM[i],yerr=dlogR[i],color=col[i],marker='o',markeredgecolor=col[i])
    pl.errorbar(logM1[i],logRe1[i],xerr=dlogM1[i],yerr=dlogR1[i],color=col[i],marker='s',markeredgecolor=col[i])
    pl.scatter(logM[i],logRe[i],label=name[i],color=col[i],marker='o',s=100,edgecolor='none')
    pl.scatter(logM1[i],logRe1[i],color=col[i],marker='s',s=100,edgecolor='none')
    #plot_ellipses(logM1[i],logRe1[i],dlogM1[i],dlogR1[i],rho[i],col=col[i])
    #plot_ellipses(np.array(logM[i]),logRe[i],dlogM[i],dlogR[i],rho[i],col=col[i])
pl.legend(loc='upper left')
pl.ylabel('$\log(R_e/kpc)$')
pl.xlabel('$\log(M/M_{\odot})$')


logRe,logM,dlogR,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_lens_2src_new.npy').T
logRe1,logM1,dlogR1,dlogM1,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_lens_1src_new.npy').T
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']

col = ['Navy','Yellow','Crimson','SteelBlue','CornflowerBlue','SeaGreen','Cyan','LightPink','Gray','BlueViolet','Chartreuse','DarkSalmon','DarkGoldenRod']
pl.figure()
for i in range(name.size):
    pl.errorbar(logM[i],logRe[i],xerr=dlogM[i],yerr=dlogR[i],color=col[i],marker='o',markeredgecolor=col[i])
    pl.errorbar(logM1[i],logRe1[i],xerr=dlogM1[i],yerr=dlogR1[i],color=col[i],marker='s',markeredgecolor=col[i])
    pl.scatter(logM[i],logRe[i],label=name[i],color=col[i],marker='o',s=100,edgecolor='none')
    pl.scatter(logM1[i],logRe1[i],color=col[i],marker='s',s=100,edgecolor='none')
pl.legend(loc='upper left')
pl.ylabel('$\log(R_e/kpc)$')
pl.xlabel('$\log(M/M_{\odot})$')


