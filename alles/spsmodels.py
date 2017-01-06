import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
from itertools import product

chabrier = np.load('/data/mauger/STELLARPOP/chabrier_Z=0.0200.dat')
l = chabrier.keys()
m = []
for i in range(len(l)):
    m.append(l[i].split('_'))

mus, tVs, ts, ages, Zs = [],[],[],[],[]
for i in range(len(l)):
    mus.append(m[i][4])
    tVs.append(m[i][3])
    ts.append(m[i][5])
    ages.append(m[i][-1])
    Zs.append(m[i][2])

mus = np.unique(mus)
tVs = np.unique(tVs)
ts = np.unique(ts)
ages = np.unique(ages)
Zs = np.unique(Zs)
                

# we don't have a wavelength axis anywhere, and I'm not sure where these masses are coming from. But ok...
# make arrays for the axes
ages = [ages[i][:9].split('=')[1] for i in range(len(ages))]
ts = [ts[i][2:] for i in range(len(ts))]
mus = [mus[i][3:] for i in range(len(mus))]
tVs = [tVs[i][3:] for i in range(len(tVs))]
Zs = np.array(['0.0001', '0.0004', '0.0040', '0.0080', '0.0200', '0.0500'])
Zgrid = Zs.astype(np.float)
#Zgrid = np.log10(Zgrid)
axes = {}
agegrid = np.array(ages).astype(np.float)
tgrid = np.array(ts).astype(np.float)
tVgrid = np.array(tVs).astype(np.float)
axes[0] = splrep(Zgrid, np.arange(Zgrid.size),k=1,s=0)
axes[1] = splrep(agegrid, np.arange(agegrid.size),k=1,s=0)
axes[2] = splrep(tgrid, np.arange(tgrid.size),k=1,s=0)
axes[3] = splrep(tVgrid, np.arange(tVgrid.size),k=1,s=0)

# now read in seds and make a grid of the magnitudes...
mags = np.zeros((Zgrid.size,agegrid.size,tgrid.size,tVgrid.size))
masses = mags*0.
## do this for the different filters
kfilt,ifilt = tools.filterfromfile('Kp_NIRC2'),tools.filterfromfile('F814W_ACS')
v5filt,v6filt = tools.filterfromfile('F555W_ACS'), tools.filterfromfile('F606W_ACS')
Gfilt,Rfilt = tools.filterfromfile('g_SDSS'), tools.filterfromfile('r_SDSS')
Ifilt,Zfilt = tools.filterfromfile('i_SDSS'), tools.filterfromfile('z_SDSS')
filters = [kfilt, ifilt, v5filt, v6filt, Gfilt, Rfilt, Ifilt, Zfilt]
mag = [mags]*7

## all metallicities
chabs = ['/data/mauger/STELLARPOP/chabrier_Z='+Zs[i]+'.dat' for i in range(Zgrid.size)]
# to simplify, shall we just do all redshifts for all bands atm?

for i,j,k in product(range(mags.shape[0]),range(mags.shape[1]),range(mags.shape[2]),range(mags.shape[3])):
    chab = chabs[i]
    sed = chab['bc03_chab_Z='+Zgrid[i]+'_tV='+tVs[k]+'_mu=0.3_t='+ts[j]+'_eps=0.000_age='+ages[i]+'0.sed']['sed']
    for filt in range(len(filters)):
        mag[filt[-2,i,j,k]] = tools.ABFM(filter[filt],[wave,sed],redshift)


