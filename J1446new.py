import pyfits as py, numpy as np, pylab as pl

# load V-band science data, cut out the lens system and plot it
V = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F606W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F606W_sci.fits')[0].header.copy()



Vcut = V[3730:3845,3690:3810]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F606W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F606W_wht.fits')[0].header.copy()
V_wht_cut = V_wht[3730:3845,3690:3810]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/J1446/F606W_sci_cutout.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1446/F606W_wht_cutout.fits',V_wht_cut,header_wht,clobber=True)


''' psfs! '''
psf1 = V[3896-14:3896+14,3509-14:3509+14]
psf2 = V[3259.5-14.5:3259.5+14.5,3888-14:3888+14]
psf3 = V[3164.5-14.5:3164.5+14.5,3408.5-14.5:3408.5+14.5]
psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)

pl.figure()
pl.imshow(psf1,interpolation='nearest')
pl.figure()
pl.imshow(psf2,interpolation='nearest')
pl.figure()
pl.imshow(psf3,interpolation='nearest')

py.writeto('/data/ljo31/Lens/J1446/F606W_psf1.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/J1446/F606W_psf2.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/J1446/F606W_psf3.fits', psf3, clobber=True)


''' I BAND '''
# load V-band science data, cut out the lens system and plot it
I = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F814W_sci.fits')[0].header.copy()
Icut = I[3730:3845,3690:3810]
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

# load I-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1446/SDSSJ1446+3856_F814W_wht.fits')[0].header.copy()
I_wht_cut = I_wht[3730:3845,3690:3810]
pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/J1446/F814W_sci_cutout.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1446/F814W_wht_cutout.fits',I_wht_cut,header_wht,clobber=True)




''' PSF '''
psf1 = I[3895-14:3895+14,3508.5-14.5:3508.5+14.5]
psf2 = I[3259-14:3259+14,3887-14:3887+14]
psf3 = I[3163.5-14.5:3163.5+14.5,3408.5-14.5:3408.5+14.5]

psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)

pl.figure()
pl.imshow(np.log10(psf1),interpolation='nearest')
pl.figure()
pl.imshow(np.log10(psf2),interpolation='nearest')
pl.figure()
pl.imshow(np.log10(psf3),interpolation='nearest')

py.writeto('/data/ljo31/Lens/J1446/F814W_psf1.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/J1446/F814W_psf2.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/J1446/F814W_psf3.fits', psf3, clobber=True)



#psf2 = I[2566.5-14.5:2566.5+14.5,2268.5-14.5:2268.5+14.5]
#psf3 = I[2602-15:2602+15,2317-15:2317+15]
#psf4 = I[494.5-14.5:494.5+14.5,1308-15:1308+15]

#psf2 = psf2/np.sum(psf2)
#psf3 = psf3/np.sum(psf3)
#psf4 =psf4/np.sum(psf4)

#pl.figure()
#pl.imshow(np.log10(psf2),interpolation='nearest')
#pl.figure()
#pl.imshow(np.log10(psf3),interpolation='nearest')
#pl.figure()
#pl.imshow(np.log10(psf3),interpolation='nearest')

#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf1.fits',psf2,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf2.fits',psf3,clobber=True)
#py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_psf3.fits',psf4,clobber=True)
