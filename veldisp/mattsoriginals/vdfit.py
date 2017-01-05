import pyfits,numpy
import tools
import special_functions as sf
from veldispfit import finddispersion
from scipy import ndimage,signal
from math import sqrt,log10,log

def pipeline(science,templates,redshift,sigmasci,sigmatmp,regions=None,nfit=7,sigma=200.,mask=None,rmask=None,omask=None,niter=None,tmodels=None,nmult=1,sdss=0,spec1d=0,smax=501.,smin=5.):
    """
    Inputs:
    science   - The name of the science file or a list containing the name of
                 the science and variance spectra.
    templates - A list containing the templates to use in the fit
    redshift  - The redshift of the galaxy
    sigmasci  - The instrumental resolution (sigma, in km/s)
    sigmatmp  - The resolution of the templates (sigma, in km/s)
    regions   - regions to fit, in restframe angstroms (eg [[4100,4500]])
    nfit      - order of the polynomial continuum
    gain      - instrumental gain of the science spectrum
    sdss      - True/[False] -> flag for inputting SDSS spectra
    sigma     - initial guess for the velocity dispersion
    mask      - a pixel-by-pixel mask (None to not use)
    nmult     - order of wavelength scaling (a la Czoske and Koopmans)
    rmask     - rest frame regions to mask (eg [[6860,6920],[7595,7030]])
    omask     - observed-frame regions to mask (eg [[4330,4350],[4850,4870]])

    Outputs:
    Dictionary including:
        chi:         chi square of best fit
        vel:         best-fit offset velocity
        sigma:       best-fit velocity dispersion
        errors:      high and low error bounds (relative to the best VD)
        model:       best-fit template model
        continuum:   model of the continuum that was fit
        science:     pixels from the science spectrum used in the fit
        wave:        wavelength of pixels used in the fit
        chisurface:  chi-square grid and minimum/maximum velocity
                        offset/dispersion values of grid
    """

    # Load in the science spectrum
    scispec,varspec,sciwave = tools.getData(science,sdss,spec1d)

    scispec[numpy.isnan(scispec)] = 0.
    varspec[numpy.isnan(varspec)] = 1e12

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # Trim edge zeros
    edges = numpy.where(scispec!=0)[0]
    start = edges[0]
    end = edges[-1]+1
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    sciwave = sciwave[start:end]


    # remove the redshift
    sciwave = sciwave/(1.+redshift)

    # prepare the templates
    ntemplates = len(templates)
    ntemp = 1
    result = []
    models = []
    t = []

    tmin = 1e9
    tmax = 0
    if regions==None:
        tmin = sciwave.min()
        tmax = sciwave.max()
    else:
        for a,b in regions:
            if a<tmin:
                tmin = a
            if b>tmax:
                tmax = b

    for template in templates:
        # Load the template
        file = pyfits.open(template)

        # Tommaso's ESI templates
        if template.split('/')[-1][:4]=='scHR':
            tmpspec = file[0].data.astype(numpy.float64)
            tmpwave = tools.wavelength(template,0)
        # INDO-US templates
        elif template[-8:]=='III.fits':
            tmpspec = file[0].data.astype(numpy.float64)
            tmpwave = tools.wavelength(template,0)
        # ELODIE templates
        else:
            tmpspec = file[1].data.astype(numpy.float64)
            tmpwave = tools.wavelength(template,1)

        tmpspec /= tmpspec.mean()

        # These steps only need to be done once
        if len(t)==0:
            # Find the best output scale
            outwave = tools.outselect(sciwave,sciwave,tmpwave)
            corrsci = tools.outselect(scispec,sciwave,tmpwave)
            corrvar = tools.outselect(varspec,sciwave,tmpwave)
            corrvar[corrvar<=0.] = 1. # Just in case.....
            if mask is not None:
                mask = tools.outselect(mask,sciwave,tmpwave)
            outwave = numpy.log10(outwave)
        # Smooth the templates to the instrumental resolution of the
        #   science data (unless the two resolutions are already
        #   similar)
        twave = numpy.log10(tmpwave)
        tmpscale = twave[1]-twave[0]
        # ~600 km/s buffer
        voff = numpy.log10(1.002)
        c = (twave+voff>numpy.log10(tmin))&(twave-voff<numpy.log10(tmax))
        twave = twave[c].copy()
        tmpspec = tmpspec[c].copy()
        if tmodels is None:
            t.append(tools.getmodel(twave,tmpspec,tmpscale,sigmasci,sigmatmp,smax=smax))
    if tmodels is not None:
        t = [i for i in tmodels]
    # send the spectra off to the fitting program!
    tmp = finddispersion(corrsci,corrvar,t,tmpwave,outwave,tmpscale,redshift,regions,nfit,sigma,mask,nmult,rmask,omask,niter=niter,smax=smax,smin=smin)

    tmp['models'] = t
    tmp['var'] = corrvar
    return tmp

    # build/return the output dictionary
    return {'vel':tmp[0],'sigma':tmp[1],'errors':tmp[2],'model':tmp[3],'continuum':tmp[4],'science':tmp[5],'wave':tmp[6],'fullsci':tmp[7],'fullwave':tmp[8],'mask':tmp[9],'solution':tmp[10],'velchain':tmp[11],'sigmachain':tmp[12],'coeffs':tmp[13],'cov':tmp[14],'models':t,'var':corrvar}
