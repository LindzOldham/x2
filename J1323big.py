import pyfits as py, numpy as np, pylab as pl

# load V-band science data, cut out the lens system and plot it
#V = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci.fits')[0].data.copy()
V = py.open('/data/mauger/EELs/SDSSJ1323+3946/F555W/SDSSJ1323+3946_F555W_sci.fits')[0].data.copy()



header = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci.fits')[0].header.copy()
Vcut = V[2050-70:2050+70,3245-70:3245+70]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_wht.fits')[0].header.copy()
V_wht_cut = V_wht[2050-70:2050+70,3245-70:3245+70]#[2050-50:2050+50,3245-50:3245+50]
#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_sci_cutout_big.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F555W_wht_cutout_big.fits',V_wht_cut,header_wht,clobber=True)


''' I BAND '''
# load V-band science data, cut out the lens system and plot it
#I = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci.fits')[0].data.copy()
I = py.open('/data/mauger/EELs/SDSSJ1323+3946/F814W/SDSSJ1323+3946_F814W_sci.fits')[0].data.copy() 
header = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci.fits')[0].header.copy()
Icut = I[2050-70:2050+70,3245-70:3245+70]#[2050-50:2050+50,3245-50:3245+50]
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_wht.fits')[0].header.copy()
I_wht_cut = I_wht[2050-70:2050+70,3245-70:3245+70]#[2050-50:2050+50,3245-50:3245+50]
pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_sci_cutout_big.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1323/SDSSJ1323+3946_F814W_wht_cutout_big.fits',I_wht_cut,header_wht,clobber=True)


