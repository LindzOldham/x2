import pyfits as py, numpy as np, pylab as pl
'''
V = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci.fits')[0].header.copy()

Vcut = V[2150-31:2150+31,2264-31:2264+31]

Vcut[Vcut<-1] = 0
pl.figure()
pl.imshow(np.log10(Vcut),origin='lower',interpolation='nearest')

V_wht = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_wht.fits')[0].data.copy()
header_wht = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_wht.fits')[0].header.copy()

V_wht_cut = V_wht[2150-31:2150+31,2264-31:2264+31]

py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_wht_cutout_med.fits',V_wht_cut,header_wht,clobber=True)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_sci_cutout_med.fits',Vcut,header,clobber=True)

# psf
psf = V[2499:2540,1965:2010]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits',psf,clobber=True)

psf = V[2520-21:2520+20, 1989-22:1989+22]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf.fits',psf,clobber=True)

psf = V[3719.5-21.5:3719.5+21.5,2400.5-21.5:2400.5+21.5]
psf = psf/np.sum(psf)
py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F555W_psf_2.fits',psf,clobber=True) # this is not a point source stupid!

'''
#pl.figure()
#pl.imshow(np.log10(psf),origin='lower',interpolation='nearest')

### I band

I = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci.fits')[0].data.copy()
header = py.open('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_sci.fits')[0].header.copy()

psfs = []
psf = I[3009-31:3009+31,2726.5-30.5:2726.5+30.5]
psfs.append(psf)
psf = I[2746-31:2746+31,3206.5-30.5:3206.5+30.5]
psfs.append(psf)
psf = I[2559-31:2559+31,4458.5-30.5:4458.5+30.5]
psfs.append(psf)
psf = I[1376-31:1376+31,2714-31:2714+31]
psfs.append(psf)
psf = I[2519.5-30.5:2519.5+30.5,1988.5-30.5:1988.5+30.5]
psfs.append(psf)
psf = I[2116-31:2116+31,1072.5-30.5:1072.5+30.5]
psfs.append(psf)
psf = I[2280.5-30.5:2280.5+30.5,3405.5-30.5:3405.5+30.5]
psfs.append(psf)
psf = I[4653-30:4653+30,2892-30:2892+30]
psfs.append(psf)
psf = I[4056-25:4056+25,3493-25:3493+25]
psfs.append(psf)
psf = I[4393-30:4393+30,4230-30:4230+30]
psfs.append(psf)

for i in range(len(psfs)):
    psfs[i] = psfs[i] / np.sum(psfs[i])
    pl.figure()
    pl.imshow(np.log10(psfs[i]),interpolation='nearest',origin='lower')
    pl.colorbar()
    pl.title('psf number '+str(i))
    print psfs[i].shape
    print np.sum(psfs[i])
    py.writeto('/data/ljo31/Lens/J1605/F814W_psf_#'+str(i)+'.fits', psfs[i],clobber=True)          










''' psf '''
#psf = I[2499:2540,1965:2010]
#psf = psf/np.sum(psf)
#py.writeto('/data/ljo31/Lens/SDSSJ1605+3811_F814W_psf.fits',psf,clobber=True)
#pl.figure()
#pl.imshow(np.log10(psf),origin='lower',interpolation='nearest')


#psf = I[2520-21:2520+20, 1989-23:1989+22]
#psf = psf/np.sum(psf)
#py.writeto('/data/ljo31/Lens/J1605/SDSSJ1605+3811_F814W_psf.fits',psf,clobber=True)
'''

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
'''
