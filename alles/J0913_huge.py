import pyfits as py, numpy as np, pylab as pl

name = 'J0913+4237'


# load V-band science data, cut out the lens system and plot it
V = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F555W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F555W_sci.fits')[0].header.copy()



Vcut = V[2400:2450,1960:2010]
Vcut = V[2380:2470,1945:2030]
Vcut = V[2330:2520,1895:2080]

#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

# load V-band weight data, cut it and plot it
V_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F555W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F555W_wht.fits')[0].header.copy()
V_wht_cut = V_wht[2400:2450,1960:2010]
V_wht_cut = V_wht[2330:2520,1895:2080]#[2380:2470,1945:2030]

#Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_sci_cutout_huge.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_wht_cutout_huge.fits',V_wht_cut,header_wht,clobber=True)


#''' psfs! '''
'''psf1 = V[4348-14:4348+14,1729-14:1729+14]
psf1 = psf1[:-2,:-2]
psf2 = V[5103.5-14.5:5103.5+14.5,3477-14:3477+14]
psf2 = psf2[:-1,:-1]
psf3 = V[1785-14:1785+14,2096-14:2096+14]
psf3 = psf3[:-2,:-2]
psf4 = V[1788.5-14.5:1788.5+14.5,1484-14:1484+14]
psf4 = psf4[:-2,:-2]
psf5 = V[2405-14:2405+14,2356-14:2356+14]
psf5 = psf5[:-2,:-2]

psf1 = psf1/np.sum(psf1)
psf2 = psf2/np.sum(psf2)
psf3 = psf3/np.sum(psf3)
psf4 = psf4/np.sum(psf4)
psf5 = psf5/np.sum(psf5)


pl.figure()
pl.imshow((psf1),interpolation='nearest')
pl.figure()
pl.imshow((psf2),interpolation='nearest')
pl.figure()
pl.imshow((psf3),interpolation='nearest')
pl.figure()
pl.imshow(psf4,interpolation='nearest')
pl.figure()
pl.imshow(psf5,interpolation='nearest')

py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_psf1.fits', psf1, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_psf2.fits', psf2, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_psf3.fits', psf3, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_psf4.fits', psf4, clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F555W_psf5.fits', psf5, clobber=True)'''


''' I BAND '''
# load V-band science data, cut out the lens system and plot it
I = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_sci.fits')[0].header.copy()
Icut = I[2400:2450,1960:2010]
Icut = I[2330:2520,1895:2080]#[2380:2470,1945:2030]

pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

# load I-band weight data, cut it and plot it
I_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/'+str(name[:5])+'/SDSS'+str(name)+'_F814W_wht.fits')[0].header.copy()
I_wht_cut = I_wht[2400:2450,1960:2010]
I_wht_cut = I_wht[2330:2520,1895:2080]#[2380:2470,1945:2030]

pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

# save both
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_sci_cutout_huge.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/'+str(name[:5])+'/F814W_wht_cutout_huge.fits',I_wht_cut,header_wht,clobber=True)




''' PSF '''
'''psf1 = I[4347.5-14.5:4347.5+14.5,1729-14:1729+14]
psf1 = psf1[:-1,:-2]
psf2 = I[1788-14:1788+14,1484-14:1484+14]
psf2 = psf2[:-1,:-2]
psf3 = I[2405-14:2405+14,2356-14:2356+14]
psf3 = psf3[:-2,:-2]
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

'''
