import pyfits as py, numpy as np, pylab as pl

# load V-band science data, cut out the lens system and plot it
#V = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci.fits')[0].data.copy()
V = py.open('/data/mauger/EELs/SDSSJ1323+3946/F555W/SDSSJ1323+3946_F555W_sci.fits')[0].data.copy()



header = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci.fits')[0].header.copy()
Vcut = V[2050-50:2050+50,3245-50:3245+50]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_wht.fits')[0].header.copy()
V_wht_cut = V_wht[2050-50:2050+50,3245-50:3245+50]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci_cutout.fits',Vcut,header,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_wht_cutout.fits',V_wht_cut,header_wht,clobber=True)

''' psfs! '''
psf = V[1657-15:1657+15,1442.5-14.5:1442.5+14.5]
psf2 = V[2603.5-14.5:2603.5+14.5,1105-15:1105+15]
psf3 = V[3005.5-14.5:3005.5+14.5,574-15:574+15]
psf = psf/np.sum(psf)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)

pl.figure()
pl.imshow(np.log10(psf),interpolation='nearest')
pl.figure()
pl.imshow(np.log10(psf2),interpolation='nearest')
pl.figure()
pl.imshow(np.log10(psf3),interpolation='nearest')

#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf1.fits',psf,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf2.fits',psf2,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_psf3.fits',psf3,clobber=True)


''' I BAND '''
# load V-band science data, cut out the lens system and plot it
#I = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci.fits')[0].data.copy()
I = py.open('/data/mauger/EELs/SDSSJ1323+3946/F814W/SDSSJ1323+3946_F814W_sci.fits')[0].data.copy() 
header = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci.fits')[0].header.copy()
Icut = I[2050-50:2050+50,3245-50:3245+50]
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_wht.fits')[0].header.copy()
I_wht_cut = I_wht[2050-50:2050+50,3245-50:3245+50]
pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci_cutout.fits',Icut,header,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_wht_cutout.fits',I_wht_cut,header_wht,clobber=True)




''' PSF '''
psf2 = I[2566.5-14.5:2566.5+14.5,2268.5-14.5:2268.5+14.5]
psf3 = I[2602-15:2602+15,2317-15:2317+15]
psf4 = I[494.5-14.5:494.5+14.5,1308-15:1308+15]

psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)
psf4 =psf4/np.sum(psf4)

pl.figure()
pl.imshow(np.log10(psf2),interpolation='nearest')
pl.figure()
pl.imshow(np.log10(psf3),interpolation='nearest')
pl.figure()
pl.imshow(np.log10(psf3),interpolation='nearest')

#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf1.fits',psf2,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf2.fits',psf3,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf3.fits',psf4,clobber=True)
