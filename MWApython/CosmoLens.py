import pylab as pl, numpy as np, pyfits as py
from scipy.interpolate import splrep,splint,splev
import itertools

''' choose a cosmology - we have already made it flat by our definition of the proper distance D_M '''
om,ol,ok,o_r = 0.3,0.7,0,0

z = np.logspace(-9,np.log10(7.5),800)
z2 = z[:-1]
zrev = z[::-1]
Ez = om*(1.+z)**3. + ol + ok*(1.+z)**2. + o_r*(1.+z)**4.
Ezrt = Ez**-0.5
Emod = splrep(z,Ezrt)

zlens = 0.306
z1 = 0.542
Ds1 = splint(z1,0,Emod)
Dls1 = splint(z1,zlens,Emod)

Ds2 = np.zeros(z.size)
Dls2 = np.zeros(z.size)
for i in range(z.size):
    Ds2[i] = splint(z[i],0,Emod)
    Dls2[i] = splint(z[i],zlens,Emod)
   

''' the quantity of interest '''
#i = source 1, j = source 2, k = lens
nu = np.zeros(z.size)
for i in range(z.size):
    nu[i] = Dls1 * Ds2[i] / (Dls2[i] * Ds1)



''' RESTRICT SPACE '''
pl.figure()
ii = np.where(z>z1)
pl.plot(z[ii],nu[ii])

