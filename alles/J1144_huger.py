import pyfits as py, numpy as np, pylab as pl

name = 'J1144+1540'


# load V-band science data, cut out the lens system and plot it
V = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].header.copy()


Vcut=V[2235:2625,1735:2125]#
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].header.copy()
#V_wht_cut = V_wht[2350:2515,1845:2010] # big
V_wht_cut = V_wht[2235:2625,1735:2125]#[2285:2575,1785:2075]#[2335:2525,1835:2025]

#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_sci_cutout_huger.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_wht_cutout_huger.fits',V_wht_cut,header_wht,clobber=True)
print Vcut.shape

''' I BAND '''
# load V-band science data, cut out the lens system and plot it
I = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].header.copy()
#Icut = I[2350:2515,1845:2010] # big
Icut = I[2235:2625,1735:2125]#[2285:2575,1785:2075]#[2335:2525,1835:2025] # small

pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

# load I-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].header.copy()
#I_wht_cut = I_wht[2350:2515,1845:2010] # big
I_wht_cut = I_wht[2235:2625,1735:2125]#[2285:2575,1785:2075]#[2335:2525,1835:2025] # small

pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_sci_cutout_huger.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_wht_cutout_huger.fits',I_wht_cut,header_wht,clobber=True)




