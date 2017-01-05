import numpy as np, pylab as pl, cPickle
from linslens.ClipResult import clipburnin
from astLib import astCalc
import pyfits as py
from tools import solarmag

def VBI(name,bbands=True):
    lp, trace,det,_ = np.load('/data/ljo31b/EELs/inference/new/huge/result_212_CHECK_'+name)
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
    if bbands:
        v = model.models[bands[name]+'_ACS'].eval(s) - 2.5*ms
    else:
        print 'assuming F606W all the time'
        v = model.models['F606W_ACS'].eval(s) - 2.5*ms
    
    return V[0], B[0], I[0]



names = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new.fits')[1].data['name']
szs = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]

names = szs.keys()
names.sort()

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_huge_new_new.fits')[1].data
names = phot['name']
re = phot['Re v']
array = np.zeros(names.size)


# now do it at z=0
filename = '/data/ljo31b/EELs/spsmodels/wide/BIV_F606W_z0_chabBC03.model'
f = open(filename,'rb')
model = cPickle.load(f)
f.close()

z = model.redshifts[0]
Bsun, Vsun, Isun, vsun = solarmag.getmag('B_Johnson',z),solarmag.getmag('V_Johnson',z),solarmag.getmag('I_Cousins',z), solarmag.getmag('F606W_ACS',z) #7.1,5.8,4.4,5.33    #5.48, 4.83, 4.08, 4.72

phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_huge_new_new.fits')[1].data
names = phot['name']
re = phot['Re v']

array3 = re*0.
array4 = array3*0.
for name in names:
    V,B,I = VBI(name,bbands=False)
    print name, '& $', '%.2f'%B, '$ & $', '%.2f'%V, '$ & $', '%.2f'%I
    array3[name==names] = V
    array4[name==names] = B

np.save('/data/ljo31/Lens/LensParams/F606W_redshift0_model_ABSMAG',array3)
np.save('/data/ljo31/Lens/LensParams/V_redshift0_model_ABSMAG',array4)

