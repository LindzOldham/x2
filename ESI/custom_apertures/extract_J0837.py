import pyfits,numpy,pylab
from scipy import ndimage,interpolate
import special_functions as sf
from spectra import spectools as st
import numpy as np,pylab as pl, pyfits as py

blue = [1500,1400,1300,1200,1100,900,600,200,0,0,0]
red = [3000,3400,3700,-1,-1,-1,-1,-1,-1,-1]
arcsecperpix = [0.120,0.127,0.134,0.137,0.144,0.149,0.153,0.158,0.163,0.168]
apsize = []

def clip(arr,nsig=3.):
    a = arr.flatten()
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<s*nsig]
        if a.size==l:
            return m,s

def extract(pref,nums,wid=1.,loc=None,wht=False):
    ''' loc allows user to choose position of aperture - give one number in a tuple for each exposure. 
    wht = True goves a Gaussian aperture 
    wid = how many sigmas wide your aperture is '''
    ospex = {} # spectrum
    ovars = {} # variance
    owave = {} # wavelength (one for each order of the echelle)
    for order in range(1,11):
        ospex[order] = []
        ovars[order] = []

    for numIndx in range(len(nums)):
        num = nums[numIndx]
        print pref,num
        d = pyfits.open('%s_%04d_bgsub.fits'%(pref,num))
        v = pyfits.open('%s_%04d_var.fits'%(pref,num))

        scales = []
        for order in range(1,11):
            B = blue[order-1]
            R = red[order-1]
            slit = d[order].data.copy()
            vslit = v[order].data.copy()
            vslit[vslit<=0.] = 1e9
            vslit[numpy.isnan(vslit)] = 1e9
            vslit[numpy.isnan(slit)] = 1e9
            h = d[order].header
            x = numpy.arange(slit.shape[1])*1.
            w = 10**(h['CRVAL1']+x*h['CD1_1']) # crval1 = the reference data coordinates corresponding to pixel 1. (NB. crpix1 = reference pixel coordinates.)  cd1_1 is degrees per pixel or something! Ie. partial derivative of first axis wrt x! Linear relation obviously. And the x-axis is in log wavelength.

            slice = numpy.median(slit[:,B:R],1) # median flux at each spatial pixel (averaged over wl)
            m,s = clip(slice)
            #pylab.figure()
            #pylab.plot(slice)
            smooth = ndimage.gaussian_filter(slice,1)
            x = numpy.arange(slice.size)*1. # spatial grid
            if loc is not None:
                pos = loc[numIndx]+9.47933-2.1065*(order-1)
                fit = numpy.array([0.,smooth[pos],pos,1.])
            else:
                fit = numpy.array([0.,smooth.max(),smooth.argmax(),1.])
            fit = sf.ngaussfit(slice,fit)[0] # doing a weird gaussian fitting thing? fitting a 1d gaussian to spectrum? nb. returns p, chi2 (so p here). I think fit is the initial guess (p is sf). So params are bg, amp, mean, sigma! Sp we can expand this to model a pair of Gaussians - one for the source and one for the lens.

            ap = np.where(abs(x-fit[2])<wid/arcsecperpix[order-1],1.,0.)
            ap = ap.repeat(slit.shape[1]).reshape(slit.shape) # have defined an aperture either 1 sigma around the peak of the fitted gaussian (top hatted) or weighting. Then replicating this all along the wavelength axis
            if order==5.:
                pl.figure()
                pl.plot(x*arcsecperpix[order-1],ap)
                pl.plot(x*arcsecperpix[order-1],slice)
                pl.plot(x*arcsecperpix[order-1],smooth)
                pl.show()

            #pylab.plot(ap*20)
            ap[vslit>=1e8] = 0.
            ap = ap/ap.sum(0)
            ap[numpy.isnan(ap)] = 0.
            slit[numpy.isnan(slit)] = 0.
#            spec = (slit.T*ap).sum(1)
#            vspec = (vslit.T*ap**2).sum(1)
            spec = (slit*ap**2).sum(0) # summing spectrum - ie. extracting it

            vspec = (vslit*ap**4).sum(0)
            vspec /= numpy.median(spec)**2


            spec /= numpy.median(spec)
            ospex[order].append(spec)
            ovars[order].append(vspec)
            owave[order] = w
            scales.append(h['CD1_1'])
            #pylab.plot(w,spec)

    # ok. So now we have a spectrum for each order of the echelle, covering different wavelength ranges...
    for order in range(1,11):
        s = ospex[order]
        for i in range(len(nums)):
            print s[i][s[i]==0].size,numpy.isnan(s[i]).sum()

    scale = 1.7e-5 # of wavelengths. NB. cd1_1 = 165e-5, which is about 0.06 arcseconds/pixel
    w0 = numpy.log10(owave[1][0])
    w1 = numpy.log10(owave[10][-1]) # total wavelength coverage
    outwave = numpy.arange(w0,w1,scale)
    outspec = numpy.zeros((outwave.size,10))*numpy.nan
    outvar = outspec.copy()


    corr = numpy.load('/data/ljo31b/EELs/esi/orderCorr.dat')
    right = None
    rb = None
    rr = None
    for order in range(1,11): # purpose is to sum exposures -- here, for each order on the echelle
        w = owave[order]
        s = w*0.
        v = w*0.
        for i in range(len(nums)):
            tmp = ndimage.median_filter(ospex[order][i],7)
            s += tmp/ovars[order][i]
            v += 1./ovars[order][i]

        if order==70:
            for i in range(len(nums)):
                pylab.plot(owave[order],ospex[order][i])
                pylab.plot(owave[order],ovars[order][i]**0.5,ls='--')
            pylab.show()
        os = s/v

        ov = 1./v
        r = ov.max()*100

        for j in range(1):
            os = ndimage.median_filter(os,5)
            s = numpy.empty((os.size,len(nums)))
            v = s.copy()
            spec = w*0.
            var = w*0.
            for i in range(len(nums)):
                s[:,i] = ospex[order][i]
                v[:,i] = ovars[order][i]

            S2N = (os-s.T).T/(v.T+ov).T**0.5 # some weird SN calculation I don't quite understand yet. os and ov are the total signal and noise summed over the three exposures, where each one has been median-filtered first; s and v are done individually per exposure.

            c = abs(S2N)<5. 

            s[~c] = numpy.nan
            v[~c] = numpy.nan
            os = numpy.nansum(s/v,1)/numpy.nansum(1./v,1)
            ov = numpy.nansum(1./v,1)**-1
            print c[s==0] # choosing only those with SN<5. Why less than? Are other ones unreliable? Cosmic rays? Now we're summing over exposures. So was that to eradicate questionable pixels, ie. ones that are bright in only one exposure?
        spec = os
        var = ov

        w0,w1,mod = corr[order]
        mod = sf.genfunc(w,0.,mod)
        spec /= mod
        var /= mod**2 # what is this model that we're dividing the spectrum by? 

        c = numpy.isnan(spec)
        spec[c] = 0.
        var[c] = 1e9

        c = (w>w0)&(w<w1)

        w = w[c]
        spec = spec[c]
        var = var[c]
        if right is not None:
            left = numpy.median(spec[(w>rb)&(w<rr)])
            spec *= right/left
            var *= (right/left)**2
        try:
            rb = owave[order+1][0] # blue end is start of next order
            rr = w[-1] # red end is end of this spectrum
            right = numpy.median(spec[(w>rb)&(w<rr)]) # dunno what this is for.
        except:
            pass

        lw = numpy.log10(w)
        c = (outwave>=lw[0])&(outwave<=lw[-1])
        mod = interpolate.splrep(lw,spec,k=1)
        outspec[c,order-1] = interpolate.splev(outwave[c],mod)
        mod = interpolate.splrep(lw,var,k=1)
        outvar[c,order-1] = interpolate.splev(outwave[c],mod)
        
    spec = numpy.nansum(outspec/outvar,1)/numpy.nansum(1./outvar,1)

    var = numpy.nansum(1./outvar,1)**-1
    numpy.save('/data/ljo31b/EELs/esi/kinematics/J0837_apertures',apsize)
    # summing over orders, getting rid of where the spline representation hasn't worked (ie. has given nans)
    return outwave,spec,var
    pylab.plot(10**(outwave),spec/var**0.5)
    pylab.show()

from scipy import ndimage
import indexTricks as iT
import sys
wid = float(sys.argv[1])
name='J0837'

#pylab.figure()
ow,s,v = extract('/data/ljo31b/EELs/esi/J0837/EEL_J0837+0801',[33,34,35],loc=[57,92,25],wid=wid)

ow,s,v=ow[:-33],s[:-33],v[:-33]
pylab.figure()
pylab.subplot(211)
pylab.plot(10**ow,s,'k')
pylab.xlim(4000.,9000.)
pylab.ylim([-0.5,1])
pylab.subplot(212)
pylab.plot(10**ow,v,'r')
pylab.xlim(4000.,9000.)
pylab.ylim([0,0.05])
pylab.suptitle('J0837')
pylab.xlabel('observed wavelength')

py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/'+name+'_ap_'+str(wid)+'_spec.fits',s,clobber=True)
py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/'+name+'_ap_'+str(wid)+'_var.fits',v,clobber=True)
py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/'+name+'_ap_'+str(wid)+'_wl.fits',ow,clobber=True)
pl.show()
