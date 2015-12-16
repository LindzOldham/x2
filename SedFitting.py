import numpy as np
import pylab as pl
import pyfits as py
from stellarpop import tools

''' Predict the offset as a function of age - currently from luminosity evolution alone. Assuming a single redshift of the source population of z=0.55, which I reckon is about the average (but haven't calculated...! How lazy.) '''
ages = ['0.001','0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','3.500','4.500','5.500','6.000','7.000','8.000','9.000','10.00','12.00','15.00']
age_array = np.array([0.001,0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,3.500,4.500,5.500,6.000,7.000,8.000,9.000,10.00,12.00,15.00])
z=0.55
bfilt1 = tools.filterfromfile('F555W_ACS')
bfilt = tools.filterfromfile('F606W_ACS')
rfilt = tools.filterfromfile('F814W_ACS')
bmags_0 = np.zeros(len(ages))
rmags_0 = bmags_0*0.
bmags_z,rmags_z = bmags_0*0.,bmags_0*0.
for i in range(len(ages)):
    sed = tools.getSED('BC_Z=1.0_age='+str(ages[i])+'gyr')
    bmags_z[i] = tools.ABFM(bfilt,sed,z)
    rmags_z[i] = tools.ABFM(rfilt,sed,z)
    bmags_0[i] = tools.ABFM(bfilt,sed,0)
    rmags_0[i] = tools.ABFM(rfilt,sed,0)
'''
pl.figure()
pl.plot(age_array,bmags_0,'Navy',label='b, z=0')
pl.plot(age_array,bmags_z,'CornflowerBlue',label='b,z=0.55')
pl.plot(age_array,rmags_0,'Crimson',label='r, z=0')
pl.plot(age_array,rmags_z,'LightCoral',label='r,z=0.55')
pl.legend(loc='lower right')
pl.xlabel('age / Gyr')
pl.ylabel('AB magnitude')
'''
''' now get the evolution - so the offset between z=0.55 and z=0 - and interpolate as a function of age '''
from scipy.interpolate import splrep, splev, splint
bev = bmags_z - bmags_0
rev = rmags_z - rmags_0
pl.figure()
pl.plot(age_array,bev,'Navy',label='b')
pl.plot(age_array,rev,'Crimson',label='r')
pl.legend(loc='lower right')
pl.xlabel('age / Gyr')
pl.ylabel('magnitude evolution')
bmod = splrep(age_array,bev)
rmod = splrep(age_array,rev)
pl.plot(age_array, splev(age_array,bmod),'k--')
pl.plot(age_array,splev(age_array,rmod),'k--')

# original Hamabe & Kormendy (1987) relation: used the V band and had mue = 19.48 + 2.92 log(Re). But for <mue> subtract 1.4 for Sersic fits
# La Barbera (2003): Gunn r band for Coma: 18.68 (z=0.024), slope dependent on fitting method but 3.9, 2.5 or 3.01! 
intercept_kormendy = 19.08
#gunnr = tools.filterfromfile('r_gunn')
#offset = tools.ABFM(r_gunn # this is going to be more complicated
intercept_ljo = 18.65
