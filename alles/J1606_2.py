import pyfits as py, numpy as np, pylab as pl

V = py.open('SDSSJ1606+2235_F606W_sci.fits')[0].data.copy()
header = py.open('SDSSJ1606+2235_F606W_sci.fits')[0].header.copy()

Vcut = V[2072-105:2072+105,2279-105:2279+105]
#Vcut = V[2015:2135,2225:2315]
Vcut = V[2072-35:2072+35,2279-35:2279+35]

Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

V_wht = py.open('SDSSJ1606+2235_F606W_wht.fits')[0].data.copy()
header_wht = py.open('SDSSJ1606+2235_F606W_wht.fits')[0].header.copy()

V_wht_cut = V_wht[2072-105:2072+105,2279-105:2279+105]
#V_wht_cut = V_wht[2015:2135,2225:2315]
V_wht_cut = V_wht[2072-35:2072+35,2279-35:2279+35]

pl.figure()
pl.imshow(np.log10(V_wht_cut),origin='lower',interpolation='nearest')

py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F606W_wht_cutout_2.fits',V_wht_cut,header_wht,clobber=True)
py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F606W_sci_cutout_2.fits',Vcut,header,clobber=True)

#''' psf '''
#psf = V[1345:1395,2475:2520]
#psf = psf/np.sum(psf)
#py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F606W_psf.fits',psf,clobber=True)

#pl.figure()
#pl.imshow(np.log10(psf),origin='lower',interpolation='nearest')

### I band

I = py.open('SDSSJ1606+2235_F814W_sci.fits')[0].data.copy()
header = py.open('SDSSJ1606+2235_F814W_sci.fits')[0].header.copy()

#Icut = I[2072-105:2072+105,2279-105:2279+105]
#Icut = I[2015:2135,2225:2315]
Icut = I[2072-35:2072+35,2279-35:2279+35]

Icut[Icut<-1] = 0
pl.figure()
pl.imshow(np.log10(Icut),origin='lower',interpolation='nearest')

I_wht = py.open('SDSSJ1606+2235_F814W_wht.fits')[0].data.copy()
header_wht = py.open('SDSSJ1606+2235_F814W_wht.fits')[0].header.copy()

#I_wht_cut = I_wht[2072-105:2072+105,2279-105:2279+105]
#I_wht_cut = I_wht[2015:2135,2225:2315]
I_wht_cut = I_wht[2072-35:2072+35,2279-35:2279+35]

pl.figure()
pl.imshow(np.log10(I_wht_cut),origin='lower',interpolation='nearest')

py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F814W_wht_cutout_2.fits',I_wht_cut,header_wht,clobber=True)
py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F814W_sci_cutout_2.fits',Icut,header,clobber=True)

#''' psf '''
#psf = I[1345:1395,2475:2520]
#psf = psf/np.sum(psf)
#py.writeto('/data/ljo31/Lens/SDSSJ1606+2235_F814W_psf.fits',psf,clobber=True)

#pl.figure()
#pl.imshow(np.log10(psf),origin='lower',interpolation='nearest')
