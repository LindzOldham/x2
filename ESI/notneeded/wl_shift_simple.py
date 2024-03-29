import pyfits as py,numpy as np,pylab as pl
from scipy import ndimage,interpolate,optimize
import special_functions as sf, indexTricks as iT
from spectra import spectools as st
import pymc
import myEmcee_blobs as myEmcee



blue = [1500,1400,1300,1200,1100,900,600,200,0,0,0]
red = [3000,3400,3700,-1,-1,-1,-1,-1,-1,-1]

# determine a constant wavelength correction. When to do this -- need to use the 2D spectrum. Then update the header to account for the shift? Yes - update CRVAL1.

def clip(arr,nsig=2.):
    a = arr.flatten()
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<s*nsig]
        if a.size==l:
            return m,s

skylines = np.loadtxt('/data/ljo31b/EELs/catalogues/skylines.dat')
dir = '/data/ljo31b/EELs/catalogues/UVES/gident_'
files = ['346','437','580L','580U','800U','860L','860U']
wls,fluxes = [],[]
for file in files:
    wl,flux = np.loadtxt(dir+file+'.dat',unpack=True,usecols=[1,2])
    wl,flux=wl[flux>100.],flux[flux>100.]
    for i in range(wl.size):
        wls.append(wl[i])
        fluxes.append(flux[i])
    
wl,flux=np.array(wls), np.array(fluxes)
skylines = wl.copy()

def ngaussFit(d,wave,mu,h1,h2,nsteps=500):
    pars = []
    sigma = pymc.Uniform('sigma ',0.01,1.)
    offset = pymc.Uniform('offset',-3,3,value=0)
    pars = [sigma,offset]
    cov = np.array([0.2,0.2])
    @pymc.observed
    def logl(value=0.,pars=pars):
        lp=0
        dlogw = h2*offset
        fracw = 10**dlogw
        shiftmu = mu/fracw
        model = np.ones((wave.size,2))
        model[:,1] = np.exp(-0.5*(wave-shiftmu.value)**2./sigma.value**2.)
        fit,chi = optimize.nnls(model,d)
        lp = -0.5*chi**2.
        return lp
    S = myEmcee.Emcee(pars+[logl],cov=cov,nthreads=10,nwalkers=10)
    S.sample(nsteps)
    result = S.result()
    lp,trace,dic,_=result
    a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,i]
    #pl.figure()
    #pl.plot(trace[:,:,-1])
    #pl.show()
    return lp,trace[a1,a2]        


def extract(pref,nums,plot=False):
    ''' here is a code to get the wavelength shift and edit the header to reflect this'''
    ospex = {} # spectrum
    ovars = {} # variance
    owave = {} # wavelength (one for each order of the echelle)
    for order in range(1,11):
        ospex[order] = []
        ovars[order] = []
    
    alloffsets = []
    for numIndx in range(1):#len(nums)):
        num = nums[numIndx]
        print pref,num
        d = py.open('%s_%04d_bgsub.fits'%(pref,num))#,mode='update')
        v = py.open('%s_%04d_var.fits'%(pref,num))#,mode='update')
        
        alloffsets = []
        for order in range(4,11):
            B,R = blue[order-1],red[order-1]
            slit = d[order].data.copy()
            vslit = v[order].data.copy()
            vslit[vslit<=0.] = 1e9
            vslit[np.isnan(vslit)] = 1e9
            vslit[np.isnan(slit)] = 1e9
            h = d[order].header
            x = np.arange(slit.shape[1])*1.
            w = 10**(h['CRVAL1']+x*h['CD1_1'])
            h1,h2 = h['CRVAL1'],h['CD1_1']
            wBR = w[B:R]
            #logwBR = np.log10(wBR)
            
            slice = np.median(vslit[5:-5,B:R],0)
            sky = skylines[(skylines>min(wBR))&(skylines<max(wBR))] 
            f = flux[(skylines>min(wBR))&(skylines<max(wBR))]
            if len(sky)>10:
                sky = sky[np.argsort(f)][-10:]
            c = [slice[abs(wBR-SKY)<2.5] for SKY in sky]
            cwl = [wBR[abs(wBR-SKY)<2.5] for SKY in sky]
            offsets = []
            for i in range(len(c)):
                lp,det = ngaussFit(c[i],cwl[i],sky[i],h1,h2,nsteps=250)
                sigma,offset = det
                dlogw = h2*offset
                fracw = 10**dlogw
                shiftmu = sky[i]/fracw
                model = np.ones((cwl[i].size,2))
                model[:,1] = np.exp(-0.5*(cwl[i]-shiftmu)**2./sigma**2.)
                fit,chi = optimize.nnls(model,c[i])
                mod = (model*fit).sum(1)
                #pl.figure()
                #pl.plot(cwl[i],c[i])
                #pl.plot(cwl[i],mod)
                #pl.axvline(sky[i],color='r')
                print order, offset, np.max(c[i])
                #pl.show()
                offsets.append(offset)
            offsets = np.array(offsets)
            alloffsets.append(offsets)
        pl.figure()
        for i in range(len(alloffsets)):
            pl.hist(alloffsets[i],bins=np.linspace(-1,1,30),alpha=0.5,histtype='stepfilled')
            m,s = clip(alloffsets[i])
            print m,s
        pl.show()
       
    np.save('/data/ljo31b/EELs/esi/J0837_offsets_simple',alloffsets)

extract('/data/ljo31b/EELs/esi/J0837/EEL_J0837+0801',[33,34,35],plot=False)

