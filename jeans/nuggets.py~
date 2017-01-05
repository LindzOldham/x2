from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects


def run(k,dist=False,ap=False,apdist=True):
    # construct Sersic model for source light
    struct = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')[0]
    phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
    sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
    ZPs = dict([('F555W',25.711),('F606W',26.493),('F814W',25.947)])
    bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
    names = struct.keys()
    names.sort()

    # for 211 models, this is simple. 
    scales = [0.05 * astCalc.da(sz[name][0]) * 1e3 * np.pi/180./3600. for name in names]
    re = [struct[name]['Source 1 re'] for name in names]
    n = [struct[name]['Source 1 n'] for name in names]
    q = [struct[name]['Source 1 q'] for name in names]
    pa = [struct[name]['Source 1 pa'] for name in names]
    magv = phot['mag v']

    # make spherically symmetric EELs
    kpc2cm = 3.086e21
    gal = SBObjects.Sersic('EEL source',{'x':0,'y':0,'re':re[k]*scales[k],'n':n[k],'q':1,'pa':0,'amp':1.})
    gal.setAmpFromMag(magv[k],ZPs[bands[names[k]]])
    gal.amp *= kpc2cm**2.
    magsun = solarmag.getmag(bands[names[k]]+'_ACS',0.)
    Lsun = -0.4*(magsun - ZPs[bands[names[k]]])
    Lsun = 10**Lsun

    # Mass components
    r = np.logspace(-5,5,1501)
    sb = gal.eval(r)
    lr,light = deproject(r,sb)
    Mdm = NFW(lr,rhalo[k],Mhalo[k])
    Mstar = light2mass(lr,light,1.) *Lsun / kpc2cm**2.

    # or
    fac = Mstar[-1]/10**logM[k]
    Mstar /= fac

    # jeans model
    if dist:
        sr,sigma_dmr = veldisp(r,sb,Mdm)
        sr,sigma_starr = veldisp(r,sb,Mstar)
        vd = (sigma_dmr+sigma_starr)**0.5
        
        pl.figure()
        pl.xlabel('r / kpc')
        pl.ylabel('$\sigma$ / kms$^{-1}$')
        pl.title(names[k])
        pl.plot(sr, vd)
        pl.xlim([1e-2,200])

    if ap:
        sigma_dm = veldisp(r,sb,Mdm,ap=2.*gal.re)
        sigma_star = veldisp(r,sb,Mstar,ap=2.*gal.re)
        vd = (sigma_dm + sigma_star)**0.5
        print '&', '%.2f'%vd, r'\\'

    if ap and dist:
        pl.figtext(0.15,0.8,'$\sigma = $'+'%.2f'%vd+' kms$^{-1}$')
        pl.savefig('/data/ljo31/public_html/Lens/phys_models/2re_apertures/'+names[k]+'.pdf')
        
    if apdist:
        aps = np.logspace(-1.5,0.7,50)
        sigma_dm = veldisp(r,sb,Mdm,ap=aps*gal.re)
        sigma_star = veldisp(r,sb,Mstar,ap=aps*gal.re)        
        vd = (sigma_dm + sigma_star)**0.5
        
        pl.figure()
        pl.xlabel('aperture radius / mutiples of $r_e$')
        pl.ylabel('$\sigma$ / kms$^{-1}$')
        pl.title(names[k])
        pl.plot(aps, vd)
        pl.xlim([0,5])
        pl.savefig('/data/ljo31/public_html/Lens/phys_models/varying_apertures/'+names[k]+'.pdf')
    
masses = np.load('/data/ljo31b/EELs/inference/new/masses.npy')
logM = masses[3]
dlogM = np.mean((masses[4],masses[5]),axis=0)
Mhalo = buildhalo(10**logM)
rhalo = virialRadius(Mhalo,0.5)    

for k in range(13):
    if Mhalo[k]>15:
        continue
    run(k,apdist=True)
    
