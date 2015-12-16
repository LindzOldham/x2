import glob
import pyfits,numpy
import lensModeller
import sys

obj = sys.argv[1]
if obj[-1]=='/':
    obj = obj[:-1]
dirs = glob.glob('../../SDSS*')
dirs.sort()
if obj in dirs:
    d = obj

    z = pyfits.open('%s/z_sci.fits'%d)[0].data.copy()
    i = pyfits.open('%s/i_sci.fits'%d)[0].data.copy()
    r = pyfits.open('%s/r_sci.fits'%d)[0].data.copy()
    g = pyfits.open('%s/g_sci.fits'%d)[0].data.copy()

#b = g
#g = r
#r = (z+i)/2.

    zS = pyfits.open('%s/z_sig.fits'%d)[0].data.copy()
    iS = pyfits.open('%s/i_sig.fits'%d)[0].data.copy()
    rS = pyfits.open('%s/r_sig.fits'%d)[0].data.copy()
    gS = pyfits.open('%s/g_sig.fits'%d)[0].data.copy()

    zP = pyfits.open('%s/z_psf.fits'%d)[0].data.copy()
    iP = pyfits.open('%s/i_psf.fits'%d)[0].data.copy()
    rP = pyfits.open('%s/r_psf.fits'%d)[0].data.copy()
    gP = pyfits.open('%s/g_psf.fits'%d)[0].data.copy()

    b = [g,gS,gP]
    g = [r,rS,rP]
    r = [i,iS,iP]

    print d
    LM = lensModeller.FitLens(b,g,r)
    #LM.save('%s/manual.dat'%d)
    #k = LM.lensManager.objs.keys()[0]
    #print LM.lensManager.objs[k].pars['b']['value']

    

