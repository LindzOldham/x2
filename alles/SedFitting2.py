import numpy as np
import pylab as pl
import pyfits as py
from stellarpop import tools
from astLib import astCalc
from scipy.interpolate import splrep, splev, splint

''' Predict the offset as a function of age - currently from luminosity evolution alone. Assuming a single redshift of the source population of z=0.55, which I reckon is about the average (but haven't calculated...! How lazy.) '''
ages = ['0.001','0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','3.500','4.500','5.500','6.000','7.000','8.000','9.000','10.00','12.00','15.00','20.00']
age_array = np.array([0.001,0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,3.500,4.500,5.500,6.000,7.000,8.000,9.000,10.00,12.00,15.00,20.00])
z=0.55
tl = astCalc.tl(z)
bfilt1 = tools.filterfromfile('F555W_ACS')
bfilt = tools.filterfromfile('F606W_ACS')
rfilt = tools.filterfromfile('F814W_ACS')
bmags = np.zeros(len(ages))
rmags = bmags*0.
for i in range(len(ages)):
    sed = tools.getSED('BC_Z=1.0_age='+str(ages[i])+'gyr')
    bmags[i] = tools.ABFM(bfilt,sed,0)
    rmags[i] = tools.ABFM(rfilt,sed,0)

'''pl.figure()
pl.plot(age_array-tl,bmags,'Navy',label='b, z=0')
pl.plot(age_array,bmags,'CornflowerBlue',label='b,z=0.55')
pl.plot(age_array-tl,rmags,'Crimson',label='r, z=0')
pl.plot(age_array,rmags,'LightCoral',label='r,z=0.55')
pl.legend(loc='lower right')
pl.xlabel('age / Gyr')
pl.ylabel('AB magnitude')'''

bmodz = splrep(age_array,bmags)
bmod0 = splrep(age_array-tl,bmags)
b0,bz = splev(age_array[:-1],bmod0), splev(age_array[:-1],bmodz) # where age_array now represents the age at z=0.55
age = age_array[:-1]
'''pl.figure()
pl.plot(age,b0,'Navy',label='z=0')
pl.plot(age,bz,'CornflowerBlue',label='z=0.55')
pl.xlabel('age / Gyr')
pl.ylabel('magnitude (arbitary zeropoint)')
pl.legend(loc='lower right')
pl.figure()
pl.plot(age,b0-bz,'Navy')
pl.xlabel('age / Gyr')
pl.ylabel(r'$\Delta \mu_e$') '''
# NB age = age at z=0.55. Delta mu = b(z=0) - b(z=0.55). It's positive because b(z=0)>b(z=0.55) because the z=0 stars are less bright.
int_kormendy = 19.08
int_ljo = 18.65
diff=b0-bz
sort=np.argsort(diff)
pl.figure()
pl.plot(age,diff,'Navy')
model = splrep(diff[sort],age[sort])
ans = splev(int_kormendy-int_ljo,model)
pl.plot(ans,int_kormendy-int_ljo,marker='s',color='CornflowerBlue')
pl.xlabel('age at z =0.55 / Gyr')
pl.ylabel('offset of Kormendy relation')
#pl.title('implying an age $>$ universe age...')
print ans

'''
model = splrep(diff[::-1],age[::-1])
ans = splev(int_kormendy-int_ljo,model)
pl.plot(ans,int_kormendy-int_ljo,'s','CornflowerBlue')
print ans
'''
'''
rmod = splrep(age_array,rmags)
b0 = splev(age_array,bmod)
bz = splev
'''

'''
# original Hamabe & Kormendy (1987) relation: used the V band and had mue = 19.48 + 2.92 log(Re). But for <mue> subtract 1.4 for Sersic fits
# La Barbera (2003): Gunn r band for Coma: 18.68 (z=0.024), slope dependent on fitting method but 3.9, 2.5 or 3.01! 
intercept_kormendy = 19.08
#gunnr = tools.filterfromfile('r_gunn')
#offset = tools.ABFM(r_gunn # this is going to be more complicated
intercept_ljo = 18.65
'''
