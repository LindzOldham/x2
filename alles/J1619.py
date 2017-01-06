import pyfits as py, numpy as np, pylab as pl

name = 'J1619+2024'


# load V-band science data, cut out the lens system and plot it
V = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].header.copy()



Vcut = V[2350:2515,1885:2030] # big
#Vcut = V[2355:2500,1860:2000] # small - in future, just cut down from this on the fly

#Vcut[Vcut<-1] = 0
#pl.figure()
#pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].header.copy()
V_wht_cut = V_wht[2350:2515,1885:2030] # big

#Vcut[Vcut<-1] = 0
#pl.figure()
#pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
#py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_sci_cutout.fits',Vcut,header,clobber=True)
#py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_wht_cutout.fits',V_wht_cut,header_wht,clobber=True)


psf1 = V[761-14:761+14,2899.5-14.5:2899.5+14.5]
psf1=psf1[:-1,:-2]
psf2 = V[3896-14:3896+14,2814.5-14.5:2814.5+14.5]
psf2 = psf2[:-1,:-2]
psf3 = V[4632-14:4632+14,2473-14:2473+14]
psf3 = psf3[:-1,:-2]
psf4 = V[1773.5-14.5:1773.5+14.5,1646.5-14.5:1646.5+14.5]
psf4=psf4[:-1,:-1]
psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)
psf4 = psf4/np.sum(psf4)

pl.figure()
pl.imshow((psf1),interpolation='nearest')
pl.figure()
pl.imshow((psf2),interpolation='nearest')
pl.figure()
pl.imshow((psf3),interpolation='nearest')
pl.figure()
pl.imshow(psf4,interpolation='nearest')

py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf1.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf2.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf3.fits', psf3, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf4.fits', psf4, clobber=True)


psf1 = V[4310-14:4310+14,3369-14:3369+14]
psf1=psf1[:-1,:-1]
psf2 = V[1773-14:1773+14,1647-14:1647+14]
psf2 = psf2[:,:-2]

pl.figure()
pl.imshow((psf1),interpolation='nearest')
pl.figure()
pl.imshow((psf2),interpolation='nearest')
pl.figure()

py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf1new.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf2new.fits', psf2, clobber=True)

'''

''' I BAND '''
# load V-band science data, cut out the lens system and plot it
I = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].header.copy()
Icut = I[2350:2515,1885:2030] # big

# load I-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].header.copy()
I_wht_cut = I_wht[2350:2515,1885:2030] # big


# save both
#py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_sci_cutout.fits',Icut,header,clobber=True)
#py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_wht_cutout.fits',I_wht_cut,header_wht,clobber=True)


psf1 = I[2614-14:2614+14,813-14:813+14]
psf1 = psf1[:-1,:-1]
psf2 = I[4559-14:4559+14,3371-14:3371+14]
psf2 = psf2[:-1,:-1]
psf3 = I[2124-14:2124+14,1131-14:1131+14]
psf3 = psf3[:,:-2]
psf4 = I[4477-14:4477+14,4307-14:4307+14]
psf4 = psf4[:-1,:-1]
psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)
psf4 = psf4/np.sum(psf4)

pl.figure()
pl.imshow((psf1),interpolation='nearest')
pl.figure()
pl.imshow((psf2),interpolation='nearest')
pl.figure()
pl.imshow((psf3),interpolation='nearest')
pl.figure()
pl.imshow(psf4,interpolation='nearest')

py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf1neu.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf2neu.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf3neu.fits', psf3, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf4neu.fits', psf4, clobber=True)
'''
