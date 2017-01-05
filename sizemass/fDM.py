import numpy as np, pylab as pl, pyfits as py, cPickle

masses = np.load('/data/ljo31b/EELs/inference/new/masses_212.npy')
table = py.open('/data/ljo31/Lens/LensParams/Phot_2src_new.fits')[1].data
magv,dmagv = table['mag v'], np.min((table['mag v lo'],table['mag v hi']),axis=0)
names = table['name']
re,dre = table['re v'], np.min((table['re v lo'],table['re v hi']),axis=0)

# lens - source
logM = masses[3]
dlogM = np.mean((masses[4],masses[5]),axis=0)
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]

# solar mag to calculate L in solar luminosities in redshifted filter
from tools import solarmag
logL,dL = magv*0., magv*0.
for ii in range(magv.size):
    name = names[ii]
    logL[ii] = solarmag.mag_to_logL(magv[ii],str(bands[name])+'_ACS',sz[name][0])
    dL[ii] = logL[ii] - solarmag.mag_to_logL(magv[ii]+dmagv[ii],str(bands[name])+'_ACS',sz[name][0])
    #print L[ii],dL[ii]

logMstarL = logM - logL
MstarL = 10**logMstarL
dlogMstarL = (logM**2. + dL**2.)**0.5
# covariances? uncertainties?

# nb. we can also get virial masses!
fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results.npy')
l,m,u = fp
d = np.mean((l,u),axis=0)
dvl,dvs,dsigmal,dsigmas = d.T
vl,vs,sigmal,sigmas = m.T
sigmas,sigmal,dsigmas,dsigmal = np.delete(sigmas,6), np.delete(sigmal,6),np.delete(dsigmas,6),np.delete(dsigmal,6)
G = 4.3e-6

# now apply aperture corrections and calculate Mdyn within an effective radius
struct = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')[0]
apcorr = np.load('/data/ljo31b/EELs/esi/kinematics/aperture_corrections.npy')
sigmas /= apcorr
Mvir = magv*0.
for ii in range(magv.size):
    name = names[ii]
    n = struct[name]['Source 1 n']
    beta = 8.87 - 0.831*n + 0.0241*n**2.
    Mvir[ii] = beta*sigmas[ii]**2.*re[ii]/G
    print beta

logMvir = np.log10(Mvir)
logMvirL = logMvir - logL
MvirL = 10**logMvirL

# half-stellar mass, i.e. within the effective radius
logM -= np.log10(2.)
pl.figure()
pl.scatter(logMvir, logM,s=30,color='SteelBlue')
xline = np.linspace(10,12,20)
pl.plot(xline,xline)
pl.show()
# so there are, worryingly, three things with larger stellar masses within re than total mass within re. Hmm.
Mvir, Mstar = 10**logMvir, 10**logM

fDM = 1.-Mstar/Mvir

pl.figure()
pl.scatter(logM,fDM)
pl.show()

ii = fDM>0
