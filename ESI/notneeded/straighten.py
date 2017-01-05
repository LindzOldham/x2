#from esi import esi_pipeline
from esi import bgsub



def ngaussfit(w,d,wave,vel):
    import numpy,pymc
    from scipy import optimize
    from SampleOpt import AMAOpt

    wave = pymc.Uniform('',wave-2.,wave+2.,value=wave)
    wid = pymc.Uniform('',0.,100.,value=vel)
    @pymc.observed
    def logl(value=0.,pars=[wave,wid]):
        loc,wid = pars
        lw = wid*loc/299792.
        model = numpy.ones((w.size,2))
        model[:,1] = numpy.exp(-0.5*(w-loc)**2/lw**2)
        fit,chi = optimize.nnls(model,d)
        return -0.5*chi**2
    cov = numpy.array([0.1,5.])
    s = AMAOpt([wave,wid],[logl],[],cov=cov)
    s.sample(500)
    s = AMAOpt([wave,wid],[logl],[],cov=cov/5.)
    s.sample(500)
    s = AMAOpt([wave,wid],[logl],[],cov=cov/15.)
    s.sample(500)

    return s.trace[-1]


def straighten(dir,inname,out_prefix,cal_prefix):
    import pyfits,scipy,numpy
    from esi.biastrim import make_bias,biastrim
    from esi.straighten import startrace,straighten,fullSolution,getOrders
    import pymc
    from scipy import ndimage

    # Where things begin and end....
    blue = [1500,1400,1300,1200,1100,900,600,200,0,0,0]
    red = [3000,3400,3700,-1,-1,-1,-1,-1,-1,-1]

    readvar = 7.3

    bias = pyfits.open(cal_prefix+"_bias.fits")[0].data.astype(scipy.float32)
    bpm = pyfits.open(cal_prefix+"_bpm.fits")[0].data.astype(scipy.float32)
    flat = pyfits.open(cal_prefix+"_norm.fits")[0].data.astype(scipy.float32)

    orders,y_soln,wideorders = numpy.load(cal_prefix+"_ycor.dat")
    fullsoln = numpy.load(cal_prefix+"_full.dat")

    hdu = pyfits.open(dir+inname)
    data = hdu[0].data.copy()
    data = biastrim(data,bias,bpm)/flat
    data[flat==0] = numpy.nan
   
    if data.shape[1]<3000:
        blue = [i/2 for i in blue]
        red = [1500,1700,1850,-1,-1,-1,-1,-1,-1,-1]

    hdulist = pyfits.HDUList([pyfits.PrimaryHDU()])
    hdulist[0].header = hdu[0].header.copy()

#    slits = getOrders(data,orders,wideorders,fullsoln)
    bpm2 = ndimage.minimum_filter(numpy.where(bpm==1,1,0),3)
    slits = getOrders(data,orders,wideorders,fullsoln)
    slits2 = getOrders(bpm2,orders,wideorders,fullsoln,1)
    offsets = []
    from scipy import interpolate
    for line,hdu in [[5577.34,4],[5889.951,4],[5895.924,4],[5889.951,5],[5895.924,5],[6300.3,5],[6300.3,6],[6863.97,6],[6923.2,6],[7276.42,7],[7794.12,7],[8430.15,8],[8885.856,8],[9337.874,8],[9337.874,9]]:#,[10103.625,9],[10372.88,9],[10418.387,9]]:
        cut,wlo,whigh,disp = slits[hdu]
        x = numpy.arange(cut.shape[1])*1.
        wave = 10**(x*disp+wlo)
        c = abs(wave-line)<5.
        spec = numpy.median(cut,0)
        fit = ngaussfit(wave[c],spec[c],line,10.)
        model = interpolate.splrep(wave,x)
        x0 = interpolate.splev(line,model)
        x1 = interpolate.splev(fit[0],model)
        offsets.append(x1-x0)
    print offsets
    crpix = numpy.median(offsets)

    for indx in range(len(orders)):
        cut,wlo,whigh,disp = slits[indx]
        slit = cut.copy()
        bpm = slits2[indx][0]
        slit[bpm<0.7] = numpy.nan


        hdu = pyfits.ImageHDU(slit)
        hdu.header.update('CTYPE1','LINEAR')
        hdu.header.update('CRVAL1',wlo-crpix*disp)
        hdu.header.update('CRPIX1',1)
        hdu.header.update('CDELT1',disp)
        hdu.header.update('CTYPE2','LINEAR')
        hdu.header.update('CRVAL2',1)
        hdu.header.update('CRPIX2',1)
        hdu.header.update('CDELT2',1)
        hdu.header.update('CD1_1',disp)
        hdu.header.update('CD2_2',1)
        hdu.header.update('CD1_2',0)
        hdu.header.update('CD2_1',0)
        hdu.header.update('DC-FLAG',1)
        hdu.header.update('DISPAXIS',1)
        hdulist.append(hdu)

    hdulist.verify('fix')
    hdulist.writeto(out_prefix+"_straight.fits",clobber=True)




dir = '../raw/'
out = 'Mrk209'

import pyfits,glob
files = glob.glob('%s/ES*fits'%dir)
files.sort()
for f in files[3:]:
    h = pyfits.open(f)[0].header
    if h['TARGNAME']!='Mrk209' or h['ELAPTIME']<500:
        continue
#    if h['TARGNAME'][0] in ['D','h']:
#        continue
#    o = '%s_%s'%(h['TARGNAME'],f.split('_')[1].split('.')[0])
    o = 'Mrk209_%s'%(f.split('.')[-2])
    straighten(dir,f.split('/')[-1],o,out)


