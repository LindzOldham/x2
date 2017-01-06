import pyfits as py, numpy as np, pylab as pl

V = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci.fits')[0].header.copy()

Vcut = V[2266-38:2266+38  ,2212-36: 2212+36  ]

Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

V_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht.fits')[0].header.copy()

V_wht_cut = V_wht[2266-38:2266+38  ,2212-36: 2212+36  ]

pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci_cutout.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht_cutout.fits',V_wht_cut,header_wht,clobber=True)

''' PSF '''
#psf = V[2515-37:2515+37,2439-35:2439+35]
#psf = psf/np.sum(psf)

#pl.figure()
#pl.imshow(np.log10(psf))
#py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf.fits',psf,clobber=True) # this isn't unresolved. Need a smaller fwhm!

''' a second PSF just in case that one is on a pedestal or something '''
#psf = V[3102-37:3102+37,1928.5-34.5:1928.5+34.5]
#psf = psf/np.sum(psf)

#pl.figure()
#pl.imshow(np.log10(psf))
#py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf_#4.fits',psf,clobber=True)

''' psfs! '''
psf = V[5211-25:5211+25,2655-25:2655+25]
psf = psf/np.sum(psf)

pl.figure()
pl.imshow(np.log10(psf))
pl.colorbar()
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf.fits',psf,clobber=True)

#psf = V[4328-25:4328+25,3235.5-24.5:3235.5+24.5]
#psf = psf/np.sum(psf)

#pl.figure()
#pl.imshow(np.log10(psf))
#pl.colorbar()
#py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf_#2.fits',psf,clobber=True)

#psf = V[2243.5-24.5:2243.5+24.5,1628.5-24.5:1628.5+24.5]
#psf = psf/np.sum(psf)

#pl.figure()
#pl.imshow(np.log10(psf))
#py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_psf_#3.fits',psf,clobber=True)


''' I BAND '''
I = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci.fits')[0].header.copy()

Icut = I[2266-38:2266+38,2212-36: 2212+36]

Icut[Icut<-1] = 0
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

I_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht.fits')[0].header.copy()

I_wht_cut = I_wht[2266-38:2266+38,2212-36: 2212+36]

pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci_cutout.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht_cutout.fits',I_wht_cut,header_wht,clobber=True)

''' PSF '''
psf = I[1429-25:1429+25,2637.5-24.5:2637.5+24.5]
psf = psf/np.sum(psf)

pl.figure()
pl.imshow(np.log10(psf))
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_psf.fits',psf,clobber=True)

psf = I[2817.5-24.5:2817.5+24.5,1949-25:1949+25]
psf = psf/np.sum(psf)

pl.figure()
pl.imshow(np.log10(psf))
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_psf_#2.fits',psf,clobber=True) # use dieses???

psf = I[3613.5-24.5:3613.5+24.5,1446.5-24.5:1446.5+24.5]
psf = psf/np.sum(psf)

pl.figure()
pl.imshow(np.log10(psf))
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_psf_#3.fits',psf,clobber=True)
