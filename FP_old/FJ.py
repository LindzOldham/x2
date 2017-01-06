import pylab as pl, numpy as np, pyfits as py

logSigma, mu, logRe, dlogSigma, dmu, dlogRe = np.load('/data/ljo31b/EELs/esi/kinematics/FP_EELs_mu.npy').T
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_lens_vdfit.npy')[()]
names = sz.keys()
names.sort()

scales = np.array([astCalc.da(sz[name][0])*1e3*np.pi/180./3600. for name in names])
Re=10**logRe
dRe = dlogRe*Re
mag = mu - 2.5*np.log10(2.*np.pi*Re**2./scales**2.)
