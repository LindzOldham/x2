import scipy,numpy,cPickle
import special_functions as sf
from scipy import ndimage,optimize,signal,interpolate
from numpy import linalg
from math import sqrt,log,log10
import pymc, myEmcee_blobs as myEmcee
import numpy as np, pylab as pl

''' read in results and examine as a function of aperture! '''
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
names = sz.keys()
names.sort()
names = ['J1248','J1323','J1347','J1606','J1619','J2228']
aps = [0.75,1.0,1.25,1.5,1.75,2.0]
dic,ddic = [], []
for name in names:
    results = np.zeros((6,4))
    dresults = results*0.
    for j in range(len(aps)):
        ap = aps[j]
        result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/'+name+'_'+str(ap))
        trace = result[1]
        velL,sigL,velS,sigS = [np.median(trace[100:,:,i].ravel()) for i in range(4)]
        d1,d2,d3,d4 = [np.median(trace[100:,:,i].ravel())-np.percentile(trace[100:,:,i].ravel(),16) for i in range(4)]
        if name == 'J2228':
            velL,sigL,velS,sigS = [np.median(trace[400:,:,i].ravel()) for i in range(4)]
            d1,d2,d3,d4 = [np.median(trace[400:,:,i].ravel())-np.percentile(trace[400:,:,i].ravel(),16) for i in range(4)]
        if name == 'J1248':
            kk = np.where(trace[100,:,0]>0)
            velL,sigL,velS,sigS = [np.median(trace[100:,kk,i].ravel()) for i in range(4)]
            d1,d2,d3,d4 = [np.median(trace[100:,kk,i].ravel())-np.percentile(trace[100:,kk,i].ravel(),16) for i in range(4)]
        
        results[j] = [velL,sigL,velS,sigS]
        dresults[j] = [d1,d2,d3,d4]
    dic.append([name,results])
    ddic.append([name,dresults])

dic = dict(dic)
ddic = dict(ddic)

for key in dic.keys():
    pl.figure()
    pl.subplot(221)
    pl.suptitle(key)
    pl.plot(aps,dic[key][:,1],marker='o',c='SteelBlue')
    pl.errorbar(aps,dic[key][:,1],yerr=ddic[key][:,1],marker='o',c='SteelBlue')
    pl.xlabel('aperture radius in arcsec')
    pl.ylabel('lens dispersion')
    pl.subplot(222)
    pl.plot(aps,dic[key][:,3],marker='o',c='SteelBlue')
    pl.errorbar(aps,dic[key][:,3],yerr=ddic[key][:,3],marker='o',c='SteelBlue')
    pl.xlabel('aperture radius in arcsec')
    pl.ylabel('source dispersion')
    pl.subplot(223)
    pl.plot(aps,dic[key][:,0],marker='o',c='SteelBlue')
    pl.errorbar(aps,dic[key][:,0],yerr=ddic[key][:,0],marker='o',c='SteelBlue')
    pl.xlabel('aperture radius in arcsec')
    pl.ylabel('lens velocity')
    pl.subplot(224)
    pl.plot(aps,dic[key][:,2],marker='o',c='SteelBlue')
    pl.errorbar(aps,dic[key][:,2],yerr=ddic[key][:,2],marker='o',c='SteelBlue')
    pl.xlabel('aperture radius in arcsec')
    pl.ylabel('source velocity')
    pl.savefig('/data/ljo31/public_html/Lens/phys_models/aperture_tests/'+key+'.pdf')
    pl.close()
