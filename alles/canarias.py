import pyfits as py, numpy as np, pylab as pl
from astropy.io import fits
from astropy.wcs import WCS
from tools.simple import *


ra,dec = 15.363, -33.722
radec = np.column_stack((ra,dec))
for i in range(5,10):
    data = py.open('/data/ljo31b/lenses/c4d_140522_094404_osj_g_v1.fits')[i].data
    header = fits.open('/data/ljo31b/lenses/c4d_140522_094404_osj_g_v1.fits')[i].header
    w = WCS(header)
    coords = w.wcs_world2pix(radec,1)[0]
    print coords, data.shape
    if (0 < coords[1] <data.shape[1]) and (0 < coords[0] <data.shape[0]):
        break

'''
data = py.open('/data/ljo31b/lenses/c4d_140522_094404_osj_g_v1.fits')[i].data
header = fits.open('/data/ljo31b/lenses/c4d_140522_094404_osj_g_v1.fits')[i].header
head = py.open('/data/ljo31b/lenses/c4d_140522_094404_osj_g_v1.fits')[i].header
w = WCS(header)
coords = w.wcs_world2pix(radec,1)[0]
cutout = data[coords[1]-500:coords[1]+500,coords[0]-500:coords[0]+500]

pl.figure()
climshow(cutout)
pl.colorbar()

py.writeto('/data/ljo31b/lenses/J010127-334319_g_osj.fits',cutout,head,clobber=True)

ra,dec = 15.363, -33.722
radec = np.column_stack((ra,dec))
for i in range(5,10):
    data = py.open('/data/ljo31b/lenses/c4d_140524_094534_osj_r_v1.fits')[i].data
    header = fits.open('/data/ljo31b/lenses/c4d_140524_094534_osj_r_v1.fits')[i].header
    w = WCS(header)
    coords = w.wcs_world2pix(radec,1)[0]
    print coords, data.shape
    
    if (0 < coords[1] <data.shape[1]) and (0 < coords[0] <data.shape[0]):
        break


data = py.open('/data/ljo31b/lenses/c4d_140524_094534_osj_r_v1.fits')[i].data
header = fits.open('/data/ljo31b/lenses/c4d_140524_094534_osj_r_v1.fits')[i].header
head = py.open('/data/ljo31b/lenses/c4d_140524_094534_osj_r_v1.fits')[i].header
w = WCS(header)
coords = w.wcs_world2pix(radec,1)[0]
cutout = data[coords[1]-500:coords[1]+500,coords[0]-500:coords[0]+500]

pl.figure()
climshow(cutout)
pl.colorbar()

py.writeto('/data/ljo31b/lenses/J010127-334319_r_osj.fits',cutout,head,clobber=True)

ra,dec = 15.363, -33.722
radec = np.column_stack((ra,dec))
for i in range(5,10):
    data = py.open('/data/ljo31b/lenses/c4d_140522_094404_osi_g_v1.fits')[i].data
    header = fits.open('/data/ljo31b/lenses/c4d_140522_094404_osi_g_v1.fits')[i].header
    w = WCS(header)
    coords = w.wcs_world2pix(radec,1)[0]
    print coords, data.shape
    
    if (0 < coords[1] <data.shape[1]) and (0 < coords[0] <data.shape[0]):
        break


data = py.open('/data/ljo31b/lenses/c4d_140522_094404_osi_g_v1.fits')[i].data
header = fits.open('/data/ljo31b/lenses/c4d_140522_094404_osi_g_v1.fits')[i].header
head = py.open('/data/ljo31b/lenses/c4d_140522_094404_osi_g_v1.fits')[i].header
w = WCS(header)
coords = w.wcs_world2pix(radec,1)[0]
cutout = data[coords[1]-500:coords[1]+500,coords[0]-500:coords[0]+500]

pl.figure()
climshow(cutout)
pl.colorbar()

py.writeto('/data/ljo31b/lenses/J010127-334319_g_osi.fits',cutout,head,clobber=True)

ra,dec = 15.363, -33.722
radec = np.column_stack((ra,dec))
for i in range(5,10):
    data = py.open('/data/ljo31b/lenses/c4d_140524_094534_osi_r_v1.fits')[i].data
    header = fits.open('/data/ljo31b/lenses/c4d_140524_094534_osi_r_v1.fits')[i].header
    w = WCS(header)
    coords = w.wcs_world2pix(radec,1)[0]
    print coords, data.shape
    
    if (0 < coords[1] <data.shape[1]) and (0 < coords[0] <data.shape[0]):
        break


data = py.open('/data/ljo31b/lenses/c4d_140524_094534_osi_r_v1.fits')[i].data
header = fits.open('/data/ljo31b/lenses/c4d_140524_094534_osi_r_v1.fits')[i].header
head = py.open('/data/ljo31b/lenses/c4d_140524_094534_osi_r_v1.fits')[i].header
w = WCS(header)
coords = w.wcs_world2pix(radec,1)[0]
cutout = data[coords[1]-500:coords[1]+500,coords[0]-500:coords[0]+500]

pl.figure()
climshow(cutout)
pl.colorbar()

py.writeto('/data/ljo31b/lenses/J010127-334319_r_osi.fits',cutout,head,clobber=True)'''

''' make psfs '''
img1 = py.open('/data/ljo31b/lenses/J010127-334319_g_osj.fits')[0].data[460:-460,430:-490]
bg=np.median(img1[-10:,0:10])
img1 = py.open('/data/ljo31b/lenses/J010127-334319_g_osj.fits')[0].data-bg
img2 = py.open('/data/ljo31b/lenses/J010127-334319_r_osj.fits')[0].data[460:-460,430:-490]
bg=np.median(img2[-10:,0:10])
img2 = py.open('/data/ljo31b/lenses/J010127-334319_r_osj.fits')[0].data-bg

data = img1.copy()

dir = '/data/ljo31b/lenses/'

psf1 = data[488-20:488+20,267-20:267+20]
psf2 = data[423-20:423+20,232-20:232+20]
psf3 = data[209-20:209+20,156-20:156+20]

psf1 = psf1[:-3,:-2]
psf1/= np.sum(psf1)
psf2 /=np.sum(psf2)
psf3 = psf3[:-2,:-2]
psf3 /= np.sum(psf3)

pl.figure()
climshow(psf1)
pl.figure()
climshow(psf2)
pl.figure()
climshow(psf3)

py.writeto(dir+'g_psf1.fits',psf1,clobber=True)
py.writeto(dir+'g_psf2.fits',psf2,clobber=True)
py.writeto(dir+'g_psf3.fits',psf3,clobber=True)

data = img2.copy()
psf1 = data[590-20:590+20,311-20:311+20]
psf2 = data[682-20:682+20,125-20:125+20]
psf3 = data[253-20:253+20,711-20:711+20]

psf1 /= np.sum(psf1)
psf2 = psf2[:-2,:-3]
psf2 /= np.sum(psf2)
psf3 /= np.sum(psf3)

py.writeto(dir+'r_psf1.fits',psf1,clobber=True)
py.writeto(dir+'r_psf2.fits',psf2,clobber=True)
py.writeto(dir+'r_psf3.fits',psf3,clobber=True)

pl.figure()
climshow(psf1)
pl.figure()
climshow(psf2)
pl.figure()
climshow(psf3)


### NEW
''' make psfs '''
img1 = py.open('/data/ljo31b/lenses/chip5/imgg.fits')[0].data[460:-460,430:-490]
bg=np.median(img1[-10:,0:10])
img1 = py.open('/data/ljo31b/lenses/chip5/imgg.fits')[0].data-bg
img2 = py.open('/data/ljo31b/lenses/chip5/imgr.fits')[0].data[460:-460,430:-490]
bg=np.median(img2[-10:,0:10])
img2 = py.open('/data/ljo31b/lenses/chip5/imgr.fits')[0].data-bg

data = img1.copy()

dir = '/data/ljo31b/lenses/'

psf1 = data[488-20:488+20,267-20:267+20]
psf2 = data[423-20:423+20,232-20:232+20]
psf3 = data[209-20:209+20,156-20:156+20]

psf1 = psf1[:-2,:-1]
psf1/= np.sum(psf1)
psf2 /=np.sum(psf2)
psf3 /= np.sum(psf3)

pl.figure()
climshow(psf1)
pl.figure()
climshow(psf2)
pl.figure()
climshow(psf3)

py.writeto(dir+'g_psf1.fits',psf1,clobber=True)
py.writeto(dir+'g_psf2.fits',psf2,clobber=True)
py.writeto(dir+'g_psf3.fits',psf3,clobber=True)

data = img2.copy()
psf1 = data[590-20:590+20,311-20:311+20]
psf2 = data[682-20:682+20,125-20:125+20]
psf3 = data[253-20:253+20,711-20:711+20]

psf1 /= np.sum(psf1)
psf2 /= np.sum(psf2)
psf3 /= np.sum(psf3)

py.writeto(dir+'r_psf1.fits',psf1,clobber=True)
py.writeto(dir+'r_psf2.fits',psf2,clobber=True)
py.writeto(dir+'r_psf3.fits',psf3,clobber=True)

pl.figure()
climshow(psf1)
pl.figure()
climshow(psf2)
pl.figure()
climshow(psf3)


### I BAND
img = py.open('/data/ljo31b/lenses/chip5/imgi.fits')[0].data[460:-460,430:-490]
bg=np.median(img[-10:,0:10])
img = py.open('/data/ljo31b/lenses/chip5/imgi.fits')[0].data-bg

data = img.copy()

dir = '/data/ljo31b/lenses/'

psf1 = data[511-20:511+20,394-20:394+20]
psf2 = data[593-20:593+20,313-20:313+20]
psf3 = data[254-20:254+20,711-20:711+20]

psf1=psf1[:-1,:-1]
psf2=psf2[:-1,:-2]
psf3=psf3[:-2,:-2]

psf1/= np.sum(psf1)
psf2 /=np.sum(psf2)
psf3 /= np.sum(psf3)

pl.figure()
climshow(psf1)
pl.figure()
climshow(psf2)
pl.figure()
climshow(psf3)

py.writeto(dir+'i_psf1.fits',psf1,clobber=True)
py.writeto(dir+'i_psf2.fits',psf2,clobber=True)
py.writeto(dir+'i_psf3.fits',psf3,clobber=True)
