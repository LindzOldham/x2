import pyfits as py, numpy as np, pylab as pl

V = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci.fits')[0].header.copy()

Vcut = V[2266-125:2266+125  ,2212-125: 2212+125  ]#

Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

V_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht.fits')[0].header.copy()

V_wht_cut = V_wht[2266-125:2266+125  ,2212-125: 2212+125  ]#[2266-38:2266+38  ,2212-36: 2212+36  ]

pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_sci_cutout_huge.fits',Vcut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F606W_wht_cutout_huge.fits',V_wht_cut,header_wht,clobber=True)



''' I BAND '''
I = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci.fits')[0].header.copy()

Icut = I[2266-125:2266+125  ,2212-125: 2212+125  ]#[2266-38:2266+38,2212-36: 2212+36]

Icut[Icut<-1] = 0
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

I_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht.fits')[0].header.copy()

I_wht_cut = I_wht[2266-125:2266+125  ,2212-125: 2212+125  ]#[2266-38:2266+38,2212-36: 2212+36]

pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_sci_cutout_huge.fits',Icut,header,clobber=True)
py.writeto('/data/ljo31/Lens/J1347/SDSSJ1347-0101_F814W_wht_cutout_huge.fits',I_wht_cut,header_wht,clobber=True)

