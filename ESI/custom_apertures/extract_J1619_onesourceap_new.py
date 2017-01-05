import pyfits as py,numpy as np,pylab as pl
from scipy import ndimage,interpolate
import special_functions as sf
from spectra import spectools as st
import indexTricks as iT

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

def extract(pref,nums,wid=1.,wht=False,centroid=None,rein=None,lens=True,source=False,regal=None):
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
        d = py.open('%s_%04d_bgsub.fits'%(pref,num))
        v = py.open('%s_%04d_var.fits'%(pref,num))

        scales = []
        for order in range(1,11):
            B = blue[order-1]
            R = red[order-1]
            slit = d[order].data.copy()
            vslit = v[order].data.copy()
            vslit[vslit<=0.] = 1e9
            vslit[np.isnan(vslit)] = 1e9
            vslit[np.isnan(slit)] = 1e9
            h = d[order].header
            x = np.arange(slit.shape[1])*1.
            w = 10**(h['CRVAL1']+x*h['CD1_1']) # crval1 = the reference data coordinates corresponding to pixel 1. (NB. crpix1 = reference pixel coordinates.)  cd1_1 is degrees per pixel or something! Ie. partial derivative of first axis wrt x! Linear relation obviously. And the x-axis is in log wavelength.

            slice = np.median(slit[:,B:R],1) # median flux at each spatial pixel (averaged over wl)
            m,s = clip(slice)
            #pylab.figure()
            #pylab.plot(slice)
            smooth = ndimage.gaussian_filter(slice,1)
            x = np.arange(slice.size)*1. # spatial grid
            fit = np.array([0.,smooth.max(),smooth.argmax(),1.])
            fit = sf.ngaussfit(slice,fit)[0] # nb. returns p, chi2 (so p here). fit is the initial guess (p is sf). So params are bg, amp, mean, sigma! Sp we can expand this to model a pair of Gaussians - one for the source and one for the lens.
            
            lenscent = fit[2] + centroid*0.05/arcsecperpix[order-1]
            source2cent = lenscent+15.

            lensap = np.where(abs(x-lenscent)<wid/arcsecperpix[order-1],1.,0.)
            sourceap = np.where(abs(x-source2cent)<wid/arcsecperpix[order-1],1.,0.)

            if lens:
                ap = lensap
            elif source:
                ap = sourceap

            ap = ap.repeat(slit.shape[1]).reshape(slit.shape) # have defined an aperture either 1 sigma around the peak of the fitted gaussian (top hatted) or weighting. Then replicating this all along the wavelength axis
            #pylab.plot(ap*20)
            if order>30.:
                pl.figure()
                if lens:
                    pl.plot(x,lensap)
                elif source:
                    pl.plot(x,sourceap)
                pl.plot(x,slice)
                #pl.plot(x,smooth)
                pl.show()

            ap[vslit>=1e8] = 0.
            ap = ap/ap.sum(0)
            ap[np.isnan(ap)] = 0.
            slit[np.isnan(slit)] = 0.

            spec = (slit*ap**2).sum(0) # summing spectrum - ie. extracting it
            vspec = (vslit*ap**4).sum(0)
            vspec /= np.median(spec)**2
            spec /= np.median(spec)
            
            ospex[order].append(spec)
            ovars[order].append(vspec)
            owave[order] = w
            scales.append(h['CD1_1'])
            #pylab.plot(w,spec)

    # ok. So now we have a spectrum for each order of the echelle, covering different wavelength ranges...
    scale = 1.7e-5 # of wavelengths. NB. cd1_1 = 165e-5, which is about 0.06 arcseconds/pixel
    w0 = np.log10(owave[1][0])
    w1 = np.log10(owave[10][-1]) # total wavelength coverage
    outwave = np.arange(w0,w1,scale)
    outspec = np.zeros((outwave.size,10))*np.nan
    outvar = outspec.copy()

    corr = np.load('/data/ljo31b/EELs/esi/raw/may2014/run/orderCorr_BD284211.dat')
    right = None
    rb = None
    rr = None
    for order in range(2,11): # purpose is to sum exposures -- here, for each order on the echelle
        w = owave[order]
        s = w*0.
        v = w*0.
        for i in range(len(nums)):
            tmp = ndimage.median_filter(ospex[order][i],7)
            s += tmp/ovars[order][i]
            v += 1./ovars[order][i]

        os = s/v
        ov = 1./v
        r = ov.max()*100

        for j in range(1):
            os = ndimage.median_filter(os,5)
            s = np.empty((os.size,len(nums)))
            v = s.copy()
            spec = w*0.
            var = w*0.
            for i in range(len(nums)):
                s[:,i] = ospex[order][i]
                v[:,i] = ovars[order][i]

            S2N = (os-s.T).T/(v.T+ov).T**0.5# os and ov are the total signal and noise summed over the three exposures, where each one has been median-filtered first; s and v are done individually per exposure.

            c = abs(S2N)<5. 
            s[~c] = np.nan
            v[~c] = np.nan
            os = np.nansum(s/v,1)/np.nansum(1./v,1)
            ov = np.nansum(1./v,1)**-1

           
        spec = os
        var = ov
        w0,w1,mod = corr[order]
        mod = sf.genfunc(w,0.,mod)
        spec /= mod
        var /= mod**2 

        c = np.isnan(spec)
        spec[c] = 0.
        var[c] = 1e9

        c = (w>w0)&(w<w1)
        w = w[c]
        print 'w',w
        spec = spec[c]
        var = var[c]
        #print right
        if right is not None:
            left = np.median(spec[(w>rb)&(w<rr)])
            spec *= right/left
            var *= (right/left)**2
        try:
            rb = owave[order+1][0] # blue end is start of next order
            print owave[order+1]
            rr = w[-1] # red end is end of this spectrum
            print 'rb,rr',rb, rr
            right = np.median(spec[(w>rb)&(w<rr)]) 
        except:
            pass
        print 'RIGHT',right
        try:
            print 'LEFT',left
        except:
            pass
        print '---'
        lw = np.log10(w)
        c = (outwave>=lw[0])&(outwave<=lw[-1])
        mod = interpolate.splrep(lw,spec,k=1)
        outspec[c,order-1] = interpolate.splev(outwave[c],mod)
        mod = interpolate.splrep(lw,var,k=1)
        outvar[c,order-1] = interpolate.splev(outwave[c],mod)
        #pylab.plot(10**outwave,outspec[:,order-1])
        #pylab.plot(10**outwave,outvar[:,order-1]**0.5,ls='--')
    #pylab.show()
    #print outspec.shape
    #df
    spec = np.nansum(outspec/outvar,1)/np.nansum(1./outvar,1)

    var = np.nansum(1./outvar,1)**-1
    # summing over orders, getting rid of where the spline representation hasn't worked (ie. has given nans)
    ow,s,v = outwave,spec,var
    pl.figure()
    pl.subplot(211)
    pl.plot(10**ow,s,'k')
    pl.xlim(4000.,9000.)
    pl.ylim([-0.5,1])
    pl.subplot(212)
    pl.plot(10**ow,v,'r')
    pl.xlim(4000.,9000.)
    pl.ylim([0,0.05])
    pl.xlabel('observed wavelength')
    pl.suptitle(name)
    #if lens:
    #    pl.savefig('/data/ljo31/public_html/Lens/esi/apertures/final/'+name+'_'+str(wid)+'_lens.png')
    #elif source:
    #    pl.savefig('/data/ljo31/public_html/Lens/esi/apertures/final/'+name+'_'+str(wid)+'_source.png')
    #pl.show()
    if lens:
        py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+'%.2f'%wid+'_spec_lens.fits',s,clobber=True)
        py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+'%.2f'%wid+'_var_lens.fits',v,clobber=True)
        py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+'%.2f'%wid+'_wl_lens.fits',ow,clobber=True)
    elif source:
        py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+'%.2f'%wid+'_spec_sourceb.fits',s,clobber=True)
        py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+'%.2f'%wid+'_var_sourceb.fits',v,clobber=True)
        py.writeto('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+'%.2f'%wid+'_wl_sourceb.fits',ow,clobber=True)
    
    return outwave,spec,var


centroids = np.load('/data/ljo31/Lens/LensParams/esi_centroids_may2014.npy')[()]
reins = np.load('/data/ljo31/Lens/LensParams/esi_reins_may2014.npy')[()]

name = 'J1619'
phot = py.open('/data/ljo31/Lens/LensParams/Phot_2src_lensgals_huge_new.fits')[1].data
re=phot['Re v']
names=phot['name']
regal = re[names==name][0]

ow,s,v = extract('/data/ljo31b/EELs/esi/raw/may2014/run/EEL_J1619+2024',[51,52,53],wid=0.30,centroid=centroids[name],rein=reins[name],lens=True,source=False,regal=regal)


pl.show()

#ow,s,v = extract('/data/ljo31b/EELs/esi/raw/may2014/run/EEL_J1619+2024',[51,52,53],wid=0.3,centroid=centroids[name],rein=reins[name],lens=True,source=False,regal=regal)

