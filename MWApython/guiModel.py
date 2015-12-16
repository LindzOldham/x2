import pyfits,numpy
from lensFit import lensModeller


# with multiple images
img1 = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_sci_cutout.fits')[0].data.copy() 
sig1 = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_noise4_cutout.fits')[0].data.copy()
psf1 = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_psf.fits')[0].data.copy()
psf1 = psf1[6:-6,6:-6]
psf1 /= psf1.sum()

img2 = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F814W_sci_cutout.fits')[0].data.copy() 
sig2 = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F814W_noise4_cutout.fits')[0].data.copy()
psf2 = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F814W_psf.fits')[0].data.copy()
psf2 = psf2[6:-6,6:-6]
psf2 /= psf2.sum()

blue = [img1,sig1,psf1]
red = [img2,sig2,psf2]
LM = lensModeller.FitLens(blue,red)

'''
# or with a single image
img = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_sci_cutout.fits')[0].data.copy() 
sig = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_noise_cutout.fits')[0].data.copy()
psf = pyfits.open('/data/ljo31/Lens/SDSSJ1606+2235_F606W_psf.fits')[0].data.copy()
psf = psf[12:-12,12:-12]
psf /= psf.sum()


LM = lensModeller.FitLens([img,sig,psf])

'''
# noise2 and nosie2
