import pyfits as py, numpy as np, pylab as pl

name = 'J0901+2027'


# load V-band science data, cut out the lens system and plot it
V = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_sci.fits')[0].header.copy()



Vcut = V[2390:2470,1650:1730]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F606W_wht.fits')[0].header.copy()
V_wht_cut = V_wht[2390:2470,1650:1730]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_sci_cutout.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_wht_cutout.fits',V_wht_cut,header_wht,clobber=True)


#''' psfs! '''
psf1 = V[3561.5-14.5:3561.5+14.5,589-14:589+14]
psf2 = V[3608-14:3608+14,3722-14:3722+14]
psf3 = V[3903-14:3903+14,4180-14:4180+14]
psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)

pl.figure()
pl.imshow((psf1),interpolation='nearest')
pl.figure()
pl.imshow((psf2),interpolation='nearest')
pl.figure()
pl.imshow((psf3),interpolation='nearest')

py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf1.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf2.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F606W_psf3.fits', psf3, clobber=True)


''' I BAND '''
# load V-band science data, cut out the lens system and plot it
I = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].header.copy()
Icut = I[2390:2470,1650:1730]
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

# load I-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].header.copy()
I_wht_cut = I_wht[2390:2470,1650:1730]
pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_sci_cutout.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_wht_cutout.fits',I_wht_cut,header_wht,clobber=True)




''' PSF '''
psf1 = I[3561.5-14.5:3561.5+14.5,589-14:589+14]
psf2 = I[3608-14:3608+14,3722-14:3722+14]
psf3 = I[3903-14:3903+14,4180-14:4180+14]
psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)

pl.figure()
pl.imshow((psf1),interpolation='nearest')
pl.figure()
pl.imshow((psf2),interpolation='nearest')
pl.figure()
pl.imshow((psf3),interpolation='nearest')

py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf1.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf2.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_psf3.fits', psf3, clobber=True)

