import numpy as np, pylab as pl, pyfits as py

names = ['J0837','J0901','J0913','J1125','J1144','J1218']

for name in names:
    scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_1.00_spec_lens.fits')[0].data
    sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_1.00_wl_lens.fits')[0].data
    varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_1.00_var_lens.fits')[0].data
    spex = py.open('/data/ljo31b/EELs/sdss/'+name+'.fits')[1].data
    sdspec = spex['flux']
    sdwave = spex['loglam']
    sdvar = spex['ivar']**-1.
    
    pl.figure()
    pl.plot(sciwave,scispec,alpha=0.5,label='esi')
    pl.plot(sdwave,sdspec/np.mean(sdspec),alpha=0.5,label='sdss')
    pl.legend(loc='lower right')
    pl.title(name)
    pl.show()
