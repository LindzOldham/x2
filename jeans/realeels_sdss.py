from tools import solarmag
import numpy as np, pylab as pl, pyfits as py
import cPickle
from makemodel import *
from astLib import astCalc
from imageSim import SBObjects
from itertools import product
from linslens import EELsModels_huge as L
from pylens import *
from scipy.interpolate import *
import indexTricks as iT

def run(Mstar,model,z,band,mu_nugg,scale):
    model.Initialise()
    XL,YL,XS,YS,RE,N,QS,PAS,B,ETA,QL,PAL,SH,SHPA = model.Ddic['Lens 1 x'],model.Ddic['Lens 1 y'],model.Ddic['Source 1 x'],model.Ddic['Source 1 y'],model.Ddic['Source 1 re'],model.Ddic['Source 1 n'],model.Ddic['Source 1 q'],model.Ddic['Source 1 pa'],model.Ddic['Lens 1 b'],model.Ddic['Lens 1 eta'],model.Ddic['Lens 1 q'],model.Ddic['Lens 1 pa'],model.Ddic['extShear'],model.Ddic['extShear PA']
    fits = model.fits
    amp = fits[0][-2]
    if model.name == 'J0837':
        amp = fits[0][-3]
    #QS=1.
    # construct nugget and normal galaxy - both have the same stellar mass and sit in identical haloes
    gal_nugg = SBObjects.Sersic('nugget',{'x':XS+XL,'y':YS+YL,'re':RE,'n':N,'q':QS,'pa':PAS,'amp':amp}) # pix
    lens = MassModels.PowerLaw('lens 1',{'x':XL,'y':YL,'b':B,'eta':ETA,'q':QL,'pa':PAL})
    shear = MassModels.ExtShear('shear',{'x':XL,'y':YL,'b':SH,'pa':SHPA})
    lenses = [lens]#,shear]

    # get halo mass at z=0.5
    Mhalo = buildhalo(10**Mstar)#-np.log10(0.7)
    #if Mhalo>15:
    #    print Mhalo
    #    Mhalo = 14.9
    #Mhalo = highz_halo_table(Mstar,0.5)
    #print Mhalo
    rhalo = virialRadius(Mhalo,z) # kpc 

    # make a grid to evaluate source and image coordinates
    yc,xc = iT.coords((900,900))*0.5 # this is just a grid IN PIXELS
    xc,yc = xc-225.,yc-225. # PIXELS
    # centre VD calculation on source

    # mass component - nugget
    r = np.logspace(-5,7,1801) # in kpc
    sb_nugg = gal_nugg.eval(r/0.05/scale)#.flatten() # in kpc
    #pl.figure()
    #pl.plot(r,sb_nugg)
    #pl.show()
    lr,light_nugg = deproject(r,sb_nugg)
    Mdm = NFW(lr,rhalo,Mhalo) # in kpc
    Mlum_nugg = light2mass(lr,light_nugg,1.)
    fac = Mlum_nugg[-1]/10**Mstar
    Mlum_nugg /= fac

    # take sigma as a function of r, then interpolate
    _,sigma_dm_nugg = veldisp(r,sb_nugg,Mdm,ap=None)
    sr,sigma_star_nugg = veldisp(r,sb_nugg,Mlum_nugg,ap=None)
    Isigma2 = sigma_dm_nugg + sigma_star_nugg
    
    # src plane:
    xl,yl = pylens.getDeflections(lenses,[xc,yc]) # PIXELS
    cs,ss = np.cos(PAS*np.pi/180.), np.sin(PAS*np.pi/180.)
    cl,sl = np.cos(PAL*np.pi/180.), np.sin(PAL*np.pi/180.)
    xp = (xl-XL-XS)*cs + (yl-YL-YS)*ss
    yp = (yl-YL-YS)*cs - (xl-XL-XS)*ss
    rs = (xp**2. *QS + yp**2. / QS)
    #rs = ((xl-XL-XS)**2. + (yl-YL-YS)**2.)**0.5
    rs *= 0.05*scale
    # interpolate onto source plane grid
    mod_Isigma2 = splrep(sr,Isigma2) # in kpc
    int_Isigma2 = splev(rs,mod_Isigma2) # in kpc
    mod_sb = splrep(sr,sb_nugg[:-300])
    int_sb1 = splev(rs,mod_sb)
    int_sb = gal_nugg.pixeval(xl,yl,csub=23)
    '''pl.figure()
    pl.imshow(int_sb1,vmax=1)
    pl.colorbar()
    pl.figure()
    pl.imshow(int_Isigma2,vmax=10000)
    pl.colorbar()
    pl.figure()
    pl.imshow((int_Isigma2/int_sb1)**0.5,vmax=700)
    pl.colorbar()
    pl.show()
    src_sigma = (int_Isigma2/int_sb)**0.5'''

    # lens plane
    #xp = (xl-XL)*cs + (yl-YL-YS)*ss
    #yp = (yl-YL)*cs - (xl-XL-XS)*ss
    #rs = (xp**2. *QS + yp**2. / QS)
    rl = ((xc-XL)**2. + (yc-YL)**2.)**0.5
    rl *= 0.05 #  arcsec in lens plane!

    #pl.figure()
    #pl.scatter(xl,yl,c=int_Isigma2,edgecolors='none',vmin=1000)
    #pl.figure()
    #pl.scatter(xl,yl,c=int_sb,edgecolors='none')
    #pl.show()

    #pl.figure()
    #pl.imshow(np.log10(int_Isigma2[400:-100,400:-100]),interpolation='nearest',origin='lower')
    #pl.colorbar()
    #pl.figure()
    #pl.imshow(np.log10(int_sb[400:-100,400:-100]),interpolation='nearest',origin='lower')
    #pl.colorbar()
    #pl.show()

    
    '''sig= (np.sum(int_Isigma2[rs<3.])/np.sum(int_sb[rs<3.]))**0.5
    sigs = []
    for ii in np.arange(0.1,10.,0.5):
        sigs.append((np.sum(int_Isigma2[rl<ii])/np.sum(int_sb[rl<ii]))**0.5)
    sig2s = []
    for ii in np.arange(0.1,10.,0.5):
        sig2s.append(veldisp(r,sb_nugg,Mdm+Mlum_nugg,ap=ii)**0.5)
    sig3s = []
    for ii in np.arange(0.1,10.,0.5):
        sig3s.append((np.sum(int_Isigma2[rs<ii])/np.sum(int_sb[rs<ii]))**0.5)
  
    pl.figure()
    pl.plot(np.arange(0.1,10.,0.5),sig2s)
    pl.plot(np.arange(0.1,10.,0.5),sigs)
    pl.plot(np.arange(0.1,10.,0.5),sig3s)
    pl.show()'''
    int_Isigma2[rl>1.5] = 0    
    #pl.figure()
    #pl.imshow(int_Isigma2/int_sb1)
    #pl.show()
    sig = (np.sum(int_Isigma2)/np.sum(int_sb1))**0.5 #* 0.5**0.5
    print sig
    
    return sig
    

masses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_211.npy')
logM = masses[3]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_lens_vdfit.npy')[()]
names = sz.keys()
names.sort()
dir = '/data/ljo31/Lens/LensModels/twoband/'
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new_new.fits')[1].data
mus = phot['mu v']

SIGS = []

for name in names:
    print name
    if name == 'J1248':
        continue
    result = np.load(dir+name+'_211')
    
    model = L.EELs(result, name)
    lm = logM[names==name]
    sig = run(lm, model,sz[name][0],bands[name],mus[names==name],astCalc.da(sz[name][0])*1e3*np.pi/180./3600.)
    SIGS.append(sig)

np.save('/data/ljo31b/EELs/SIGS_SDSS_3',SIGS)


