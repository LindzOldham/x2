import pyfits as py, numpy as np, pylab as pl
from astropy.io import fits
from astropy.wcs import WCS
from tools.simple import *
import glob

files = glob.glob('/data/ljo31b/lenses/data/whtmaps/*.fits')

ra,dec = 15.363, -33.722
radec = np.column_stack((ra,dec))
for file in files:
    for i in range(1,10):
        data = py.open(file)[i].data
        header = fits.open(file)[i].header
        w = WCS(header)
        coords = w.wcs_world2pix(radec,1)[0]
        print coords, data.shape
        if (0 < coords[1] <data.shape[1]) and (0 < coords[0] <data.shape[0]):
            head = py.open(file)[i].header
            cutout = data[coords[1]-500:coords[1]+500,coords[0]-500:coords[0]+500]
            pl.figure()
            climshow(cutout)
            pl.colorbar()
            pl.title('chip '+str(i))
            pl.show()
            py.writeto('/data/ljo31b/lenses/chip5/'+file.split('/')[-1],cutout,head,clobber=True)
            continue
'''
# now save cutouts for modelling
# but first do we need to stack them?
g,r=0.,0.
for file in files:
    header = py.open(file)[0].header
    time, band = header['EXPTIME'], header['FILTER'][0]
    if band == 'r':
        r += time
    elif band == 'g':
        g += time
    print band, time, file


print 'g', g, 'r', r

# files with longest exposure times:
#/data/ljo31b/lenses/data/tu1828889.fits -- i band -- 2160 s
#/data/ljo31b/lenses/data/tu1851844.fits -- g band -- 4800 s
#/data/ljo31b/lenses/data/tu1853594.fits -- r band -- 4500 s

imgg = py.open('/data/ljo31b/lenses/chip5/tu1851844.fits')[0].data#[460:-460,430:-490]
imgr = py.open('/data/ljo31b/lenses/chip5/tu1853594.fits')[0].data#[460:-460,430:-490]
imgi = py.open('/data/ljo31b/lenses/chip5/tu1828889.fits')[0].data#[460:-460,430:-490]

expmapi = py.open('/data/ljo31b/lenses/chip5/tu1829348.fits')[0].data#[460:-460,430:-490]
expmapg = py.open('/data/ljo31b/lenses/chip5/tu1851986.fits')[0].data#[460:-460,430:-490]
expmapr = py.open('/data/ljo31b/lenses/chip5/tu1853672.fits')[0].data#[460:-460,430:-490]

combine expmaps with noise weighting to make proper noise maps!

import colorImage
CI = colorImage.ColorImage()
img = CI.createModel(imgg,0.5*(imgg+imgr),imgr)
img = CI.createModel(imgg,imgr,imgi)

pl.figure()'''
