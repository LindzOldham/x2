import pyfits as py, numpy as np, pylab as pl

V = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci.fits')[0].header.copy()

Vcut = V[2150-46:2150+46,2264-46:2264+46]

Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

V_wht = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_wht.fits')[0].header.copy()

V_wht_cut = V_wht[2150-46:2150+46,2264-46:2264+46]

py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_wht_cutout2.fits',V_wht_cut,header_wht,clobber=True)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout2.fits',Vcut,header,clobber=True)

''' psf '''
psf = V[2499:2540,1965:2010]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/SDSSJ1605+3811_F555W_psf.fits',psf,clobber=True)

psf = V[2520-21:2520+20, 1989-22:1989+22]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits',psf,clobber=True)

psf = V[3719.5-21.5:3719.5+21.5,2400.5-21.5:2400.5+21.5]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf_2.fits',psf,clobber=True) # this is not a point source stupid!


#pl.figure()
#pl.imshow(np.log10(psf),origin='lower',interpolation='nearest')

### I band

I = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci.fits')[0].header.copy()

Icut = I[2150-46:2150+46,2264-46:2264+46]

Icut[Icut<-1] = 0
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

I_wht = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_wht.fits')[0].header.copy()

I_wht_cut = I_wht[2150-46:2150+46,2264-46:2264+46]

py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_wht_cutout2.fits',I_wht_cut,header_wht,clobber=True)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci_cutout2.fits',Icut,header,clobber=True)

''' psf '''
#psf = I[2499:2540,1965:2010]
#psf = psf/np.sum(psf)
#py.writeto('/data/ljo31/Lens/SDSSJ1605+3811_F814W_psf.fits',psf,clobber=True)
#pl.figure()
#pl.imshow(np.log10(psf),origin='lower',interpolation='nearest')


#psf = I[2520-21:2520+20, 1989-23:1989+22]
#psf = psf/np.sum(psf)
#py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_psf.fits',psf,clobber=True)


psf = I[1691-25:1691+25,1974.5-25.5:1974.5+25.5]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_psf_2.fits',psf,clobber=True) 


psf = I[1935-11:1935+11,1938-11:1938+11]
psf = psf/np.sum(psf)
pl.figure()
pl.imshow(np.log10(psf))
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_psf_3.fits',psf,clobber=True) 

psf = I[1828.5-15.5:1828.5+15.5,2008-15:2008+15]
psf = psf/np.sum(psf)
pl.figure()
pl.imshow(np.log10(psf))
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_psf_4.fits',psf,clobber=True) 


psf = I[1229.5-21.5:1229.5+21.5,2167.5-21.5:2167.5+21.5]
psf = psf/np.sum(psf)
pl.figure()
pl.imshow(np.log10(psf))
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_psf_5.fits',psf,clobber=True) # try this one
