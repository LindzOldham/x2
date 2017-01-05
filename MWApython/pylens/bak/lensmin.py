from pylens import *
from radbspline import pyfit
import numpy,pyfits
from scipy import optimize,ndimage

imagefile = ""
maskfile = ""
variancefile = ""
psffile = ""
OVERSAMP = 2 # Oversampling factor


# First load images
image = pyfits.open(imagefile)[0].data.copy()
var = pyfits.open(variancefile)[0].data.copy()
sig = var**0.5
psf = pyfits.open(psffile)[0].data.copy()
mask = pyfits.open(maskfile)[0].data.copy()

# Make the PSF kosher
psf[psf<0.] = 0.
psfTran = None     # placeholder for the FFT'd PSF
psf = ndimage.zoom(psf,OVERSAMP)
psf /= psf.sum()

# Create SIE mass model and external shear objects
sie = massmodel.PowerLaw(1.,load=False)
extshear = massmodel.ExtShear()

# Create the source model
source = profiles.Sersic()
galaxy = None # or galaxy = profiles.Sersic()


# Initial guess of model parameters
pars = []

# Define optimization function
def chi2(p,siemodel,shear,src,gal,image,sig,psf):
    global psfTran # Needs to be global because it will be modified

    siemodel.x0    = p[0]*OVERSAMP
    siemodel.y0    = p[1]*OVERSAMP
    siemodel.b     = abs(p[2])*OVERSAMP
    siemodel.q     = abs(p[3])
    siemodel.theta = p[4]

    shear.x0       = p[0]*OVERSAMP
    shear.y0       = p[1]*OVERSAMP
    shear.b        = abs(p[5])
    shear.theta    = p[6]

    src.x          = p[7]*OVERSAMP
    src.y          = p[8]*OVERSAMP
    src.q          = abs(p[9])
    src.theta      = p[10]
    src.re         = abs(p[11])*OVERSAMP
    src.amp        = abs(p[12])
    src.n          = abs(p[13])

    # If we are fitting the galaxy
    if gal is not None:
        gal.x          = p[0]*OVERSAMP
        gal.y          = p[1]*OVERSAMP
        gal.q          = abs(p[14])
        gal.theta      = p[15]
        gal.re         = abs(p[16])*OVERSAMP
        gal.amp        = abs(p[17])
        gal.n          = abs(p[18])
    

    # Return ridiculous residuals if parameters are out of bounds
    if src.q>1 or siemodel.q>1:
        return image.flatten()*0. + 1e31

    # Define coordinate grid; it's probably faster to just make this once....
    y,x = numpy.indices((image.shape[0]*OVERSAMP,image.shape[1]*OVERSAMP)).astype(numpy.float64)

    srcmodel = src.eval(x,y)
    model = pylens.lens_images([siemodel,shear],srcmodel)

    if gal is not None:
        model += gal.eval(x,y)

    # Rescale the PSF and zoom in for the convolution
    #  If you are not modifying the PSF, it's probably best to zoom outside
    #  of the optimization loop....
    # psf0 = psf.copy()
    # psf0 = psf0**p[-1]     # Used to sharpen the PSF, for example
    # psf0 = ndimage.zoom(psf0,OVERSAMP)
    # psf0 /= psf0.sum()
    psf0 = psf

    # NOTE: If you modify the PSF you DO NOT want to use the stored transform
    if psfTran is None:
        model,psfTran = convolve.convolve(model,psf0)
    else:
        model,psfTran = convolve.convolve(model,psfTran,False)

    kernel = numpy.ones((OVERSAMP,OVERSAMP))
    model = signal.convolve(model,kernel,'same')/kernel.sum()

    if gal is None:
        # Mask pixels where the source is strong
        cond = model>3*sig
        cond = cond|mask   # apply the static mask
        var = sig**2
        var[cond] = numpy.nan
        model += pyfit.imagefit(image-model,1./k,incoords=[p[0],p[1]],ntheta=[0,-1,1,-2,2],centroid=False)

    chi = ((image-model)/sig).flatten()
    # Optimize wants chi, not chi**2
    return chi

diag = None
coeffs,ier = optimize.leastsq(chi2,pars,(sie,extshear,source,galaxy,image,sig,psf),diag=diag,factor=99.,espfcn=1e-3,ftol=1e-17)

# Print optimized parameters as a python list
o = "["
for c in coeffs:
    o += "%f,"%c
o = o[:-1]+"]"
print o
