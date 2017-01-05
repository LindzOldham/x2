import numpy as np, pylab as pl, pyfits as py
import cPickle
from astLib import astCalc

name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
source_redshifts = dict([('J0837',0.6411),('J0901',0.586),('J0913',0.539),('J1125',0.689),('J1144',0.706),('J1218',0.6009),('J1248',0.528),('J1323',0.4641),('J1347',0.63),('J1446',0.585),('J1605',0.542),('J1606',0.6549),('J1619',0.6137),('J2228',0.4370)])
keckconversion = dict([('J0837',0.2),('J0901',0.2),('J0913',0.2),('J1125',0.2),('J1144',0.2),('J1218',0.6),('J1248',0.2),('J1323',0.2),('J1347',0.6),('J1446',0.2),('J1605',0.2),('J1606',0.6),('J1619',0.2),('J2228',0.6)])


name=name[1:]
rads = np.zeros((name.size,3))
lamb = np.array([600,814,2124])
cols = ['SteelBlue','Crimson','Purple','Orange','Gray','LightPink','CornflowerBlue','Navy','SeaGreen','Yellow','Turquoise']
#pl.figure()
for ii in range(name.size):
    scale = astCalc.da(source_redshifts[name[ii]])*1e3*np.pi/180./3600.
    keckscale = keckconversion[name[ii]]
    rads[ii,0] = np.load('/data/ljo31/Lens/radgrads/'+str(name[ii])+'_211_V.npy')[0].pars['re']*0.05*scale
    rads[ii,1] = np.load('/data/ljo31/Lens/radgrads/'+str(name[ii])+'_211_I.npy')[0].pars['re']*0.05*scale
    rads[ii,2] = np.load('/data/ljo31/Lens/radgrads/'+str(name[ii])+'_211_K.npy')[0].pars['re']*0.05*scale
    #pl.figure()
    #pl.scatter(lamb,rads[ii],color=cols[ii],s=100)
    #pl.errorbar(lamb,rads[ii],yerr=rads[ii]*0.05,color=cols[ii],ecolor=cols[ii],fmt='o',markersize=10)
    #pl.xscale('log')
    #pl.yscale('log')
    #pl.title(name[ii])
    #pl.xlabel('wavelength (nm)')
    #pl.ylabel('effective radius (kpc)')
    #pl.savefig('/data/ljo31/Lens/TeXstuff/radgrad'+str(name[ii])+'.pdf')
    print name[ii],rads[ii]
