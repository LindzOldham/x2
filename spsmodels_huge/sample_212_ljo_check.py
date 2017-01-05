import numpy as np, pylab as pl, cPickle
from linslens.ClipResult import clipburnin
from astLib import astCalc
import pyfits as py
from tools import solarmag

def VBI(name):
    lp, trace,det,_ = np.load('/data/ljo31b/EELs/inference/new/huge/result_212_'+name)
    a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    a2=0

    # calculate some absolute magnitudes
    L = np.array([det[key][a1,0,a3] for key in ['logtau_V lens', 'tau lens', 'age lens','logZ lens']])
    S = np.array([det[key][a1,0,a3] for key in ['logtau_V source', 'tau source', 'age source','logZ source']])
    M = [det[key][a1,0,a3] for key in ['massL', 'massS']]
    doexp = [True,False,False,True]
    doexp = np.array(doexp)==True
    L[doexp] = 10**L[doexp]
    S[doexp] = 10**S[doexp]
    l = np.atleast_2d([L])
    s = np.atleast_2d([S])
    ml,ms = M

    # ultimately, do this
    V = model.models['V_Johnson'].eval(s) - 2.5*ms
    B = model.models['B_Johnson'].eval(s) - 2.5*ms
    I = model.models['I_Cousins'].eval(s) - 2.5*ms
    v = model.models['F606W_ACS'].eval(s) - 2.5*ms
    #print 'ms',ms

    Lv = 0.4*(Vsun-V)
    Lb = 0.4*(Bsun-B)
    Li = 0.4*(Isun-I)
    Lvv = 0.4*(vsun-v)

    RE = re[name==names]
    logIv = Lv - np.log10(2.*np.pi*RE**2.)
    logIb = Lb - np.log10(2.*np.pi*RE**2.)
    logIi = Li - np.log10(2.*np.pi*RE**2.)
    logIvv = Lvv - np.log10(2.*np.pi*RE**2.)
    return V[0], B[0], I[0], Lv, Lb, Li, logIv, logIb, logIi, v[0],Lvv, logIvv



names = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new.fits')[1].data['name']
szs = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]

filename = '/data/ljo31b/EELs/spsmodels/wide/BIV_F606W_z05_chabBC03.model'
f = open(filename,'rb')
model = cPickle.load(f)
f.close()

z = model.redshifts[0]
Bsun, Vsun, Isun, vsun = solarmag.getmag('B_Johnson',z),solarmag.getmag('V_Johnson',z),solarmag.getmag('I_Cousins',z), solarmag.getmag('F606W_ACS',z) #7.1,5.8,4.4,5.33    #5.48, 4.83, 4.08, 4.72

dl = astCalc.dl(z)
DM = 5*np.log10(dl*1e6)-5.


Bsun, Vsun, Isun, vsun = Bsun+DM, Vsun+DM, Isun+DM, vsun+DM
phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_huge_new_new.fits')[1].data
names = phot['name']
re = phot['Re v']

array = np.zeros((13,9))

for name in names:
    V,B,I,Lv,Lb,Li, logIv, logIb, logIi, v, Lvv, logIvv = VBI(name)
    print name, '& $', '%.2f'%B, '$ & $', '%.2f'%V, '$ & $', '%.2f'%I, '$ &', '%.2f'%Lb, '&', '%.2f'%Lv, '&', '%.2f'%Li, '&', '%.2f'%logIb, '&', '%.2f'%logIv, '&', '%.2f'%logIi, '&', '%.2f'%v, '&', '%.2f'%Lvv, '&', '%.2f'%logIvv, r'\\'
    array[name==names] = [B,V,I,Lb,Lv,Li,logIb,logIv,logIi]
    #print array
#np.save('/data/ljo31/Lens/LensParams/BVI_photometry_212.npy',array)

