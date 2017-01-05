import numpy,pyfits
from mostools import spectools as st
from scipy import interpolate
from math import log10

light = 299792.458
ln10 = numpy.log(10.)

VGRID = 1.

"""
Get science, variance, and wavelength data from input
"""
def getData(science,sdss=0,spec1d=0):
    # SDSS spectrum
    if sdss:
        file = pyfits.open(science)
        if file[0].data is None:    # SDSS 3 format
            scispec = file[1].data.flux.astype(numpy.float64)
            varspec = file[1].data.ivar.astype(numpy.float64)
            varspec[varspec==0] = numpy.nan
            varspec = varspec**-1
            sciwave = 10**file[1].data.loglam.astype(numpy.float64)
            return scispec,varspec,sciwave
        scispec = file[0].data[0].astype(numpy.float64)
        varspec = file[0].data[2].astype(numpy.float64)
        varspec = varspec*varspec
        sciwave = st.wavelength(science,0)
        return scispec,varspec,sciwave
    # SPEC1D spectrum
    if spec1d:
        file = pyfits.open(science)[1].data
        scispec = file.FLUX.copy()
        varspec = 1./file.IVAR
        sciwave = file.LAMBDA.copy()
        return scispec,varspec,sciwave
    # A filename for a file in MWA format
    if type(science)!=type([]):
        file = pyfits.open(science)
        scispec = file[1].data.astype(numpy.float64)
        varspec = file[3].data.astype(numpy.float64)
        sciwave = st.wavelength(science,1)
        return scispec,varspec,sciwave
    # One file containing the spectrum, another with the variance
    if len(science)==2:
        # FITS files
        try:
            scispec = pyfits.open(science[0])[0].data.astype(numpy.float64)
            varspec = pyfits.open(science[1])[0].data.astype(numpy.float64)
            sciwave = st.wavelength(science[0])
        # text files
        except:
            d = numpy.loadtxt(science[0])
            scispec = d[:,2].copy()
            sciwave = d[:,0].copy()
            varspec = numpy.loadtxt(science[1])[:,2]**2
    # One file each for spectrum, variance, and wavelength
    elif len(science)==3:
        # FITS files
        if type(science[0])==type('a'):
            scispec = pyfits.open(science[0])[0].data.astype(numpy.float64)
            varspec = pyfits.open(science[1])[0].data.astype(numpy.float64)
            sciwave = pyfits.open(science[2])[0].data.astype(numpy.float64)
        # arrays
        else:
            sciwave = science[0].copy()
            scispec = science[1].copy()
            varspec = science[2].copy()
    return scispec,varspec,sciwave

"""
Find the lowest and highest common wavelengths between the template and the
  science spectrum and return the sub-spectrum that matches the output
wavelength range
"""
def outselect(input,swave,twave):
    return input[(swave>twave.min())&(swave<twave.max())]

wavelength = st.wavelength

def getmodel(twave,tspec,tscale,sigsci,sigtmp,smin=5.,smax=501):
    from scipy import ndimage,interpolate
    import ndinterp,numpy
    match = tspec.copy()
    if sigsci>sigtmp:
        kernel = (sigsci**2-sigtmp**2)**0.5/(299792.*ln10*tscale)
        if kernel>0.2:
            match = ndimage.gaussian_filter1d(tspec.copy(),kernel)
    disps = numpy.arange(smin,smax,VGRID)
    cube = numpy.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        kernel = disp/(299792.*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = numpy.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = numpy.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)


""" Plot output from pipeline in a nice way """
def plot(answer,continuum=True,ofile=None):
    def sigclip(arr,nsig=4.):
        a = arr.copy()
        m,s,l = a.mean(),a.std(),a.size

        while 1:
            a = a[abs(a-m)<nsig*s]
            if a.size==l:
                return m,s
            m,s,l = a.mean(),a.std(),a.size

    import pylab
    pylab.clf()
    cond = (answer['fullwave']>=answer['wave'][0])&(answer['fullwave']<=answer['wave'][-1])
    cond2 = numpy.isfinite(answer['science'])

    addterm = answer['fullsci'][cond].min()-(answer['fullsci'][cond]-answer['model']).max()

    resid = pylab.axes([0.1,0.1,0.8,0.2])
    pylab.plot(answer['wave'],answer['fullsci'][cond]-answer['model'],c='k')
    pylab.xlabel('Wavelength ($\AA$)')
    pylab.xlim(answer['wave'][0],answer['wave'][-1])
    lim = pylab.xlim()
    r = answer['fullsci'][cond]-answer['model']
    r = r[numpy.isfinite(r)]
    m,s = sigclip(r)
    pylab.ylim(m-5*s,m+5*s)

    args = numpy.where(numpy.isnan(answer['science']))[0]
    if args.size>0:
        start = args[0]
#        end = args[-1]
        for i in range(1,args.size):
            if args[i]-args[i-1]>1:
                end = args[i-1]
                pylab.axvspan(answer['wave'][start],answer['wave'][end+1],fc='#AAAAAA',ec='#AAAAAA',alpha=0.5)
                start = args[i]
        end = args[-1]
        if end==answer['wave'].size-1:
            end -= 1
        pylab.axvspan(answer['wave'][start],answer['wave'][end+1],fc='#AAAAAA',ec='#AAAAAA',alpha=0.5)
    pylab.xlim(lim)

    spec = pylab.axes([0.1,0.3,0.8,0.6])
    pylab.plot(answer['wave'],answer['fullsci'][cond],c='k')
    pylab.plot(answer['wave'],answer['model'],c='r',lw=2)
    mod = answer['model'].mean()-answer['continuum'].mean()
    mod = 0.
    if continuum==True:
        pylab.plot(answer['wave'],answer['continuum']+mod,c='g')

    pylab.ylabel('Flux (arbitrary units)')
    pylab.xlim(lim)
    spec.xaxis.set_major_formatter(pylab.NullFormatter())
    ticks = spec.get_yticks()
    spec.set_yticks(ticks[1:])

    args = numpy.where(numpy.isnan(answer['science']))[0]
    if args.size>0:
        start = args[0]
#        end = args[-1]
        for i in range(1,args.size):
            if args[i]-args[i-1]>1:
                end = args[i-1]
                pylab.axvspan(answer['wave'][start],answer['wave'][end+1],fc='#AAAAAA',ec='#AAAAAA',alpha=0.5)
                start = args[i]
        end = args[-1]
        if end==answer['wave'].size-1:
            end -= 1
        pylab.axvspan(answer['wave'][start],answer['wave'][end+1],fc='#AAAAAA',ec='#AAAAAA',alpha=0.5)
    pylab.xlim(lim)

    if ofile is None:
        pylab.show()
    else:
        pylab.savefig(ofile)


def showContours(result):
    import pylab,mattplotlib

    mattplotlib.pdf2d(result['sigmachain'],result['velchain'])
    pylab.title('Delta Chi-square Contours')
    pylab.xlabel('Velocity Dispersion (km/s)')
    pylab.ylabel('Velocity offset (km/s)')

    pylab.show()
    """
    chiout = result['chisurface']

    pylab.contour(chiout[0],[1.,2.,4.,8.,16.],extent=chiout[1:])
    pylab.title('Delta Chi-square Contours')
    pylab.xlabel('Velocity Dispersion (km/s)')
    pylab.ylabel('Velocity offset (km/s)')

    pylab.show()
    """

def showVel(result):
    import pylab
    samples = result['velchain']
    nbins = samples.size/30
    if nbins>101:
        nbins = 101
    pylab.hist(samples,nbins)
    pylab.title('Velocity Posterior')
    pylab.xlabel('Velocity offset (km/s)')
    pylab.show()

def showSigma(result):
    import pylab
    samples = result['sigmachain']
    nbins = samples.size/30
    if nbins>101:
        nbins = 101
    pylab.hist(samples,nbins)
    pylab.title('Velocity Dispersion Posterior')
    pylab.xlabel('Velocity Dispersion (km/s)')
    pylab.show()

