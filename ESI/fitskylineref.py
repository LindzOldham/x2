import pyfits as py,numpy as np,pylab as pl
from scipy import ndimage,interpolate,optimize
import special_functions as sf, indexTricks as iT
from spectra import spectools as st
import pymc
import myEmcee_blobs as myEmcee

ind = [2000,7000]

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

def ngaussFit(d,wave,mu,h1,h2,nsteps=300):
    pars = []
    sigma = pymc.Uniform('sigma ',0.01,50.)
    offset = pymc.Uniform('offset',-10,10,value=0)
    pars = [sigma,offset]
    cov = np.array([0.2,0.2])
    @pymc.observed
    def logl(value=0.,pars=pars):
        lp=0
        dw = h2*offset
        shiftmu = mu+dw
        model = np.ones((wave.size,2))
        model[:,1] = np.exp(-0.5*(wave-shiftmu.value)**2./sigma.value**2.)
        fit,chi = optimize.nnls(model,d)
        lp = -0.5*chi**2.
        return lp
    S = myEmcee.Emcee(pars+[logl],cov=cov,nthreads=1,nwalkers=10)
    S.sample(nsteps)
    result = S.result()
    lp,trace,dic,_=result
    a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,i]
    #pl.figure()
    #pl.plot(trace[:,:,-1])
    #pl.show()
    #pl.figure()
    #pl.plot(lp)
    #pl.show()
    return lp,trace[a1,a2]        


def extract(pref,nums,plot=False):
    ''' here is a code to get the wavelength shift and edit the header to reflect this'''
    ospex = [] # spectrum
    ovars = [] # variance
    owave = [] # wavelength
    
    alloffsets = []
    for numIndx in range(len(nums)):
        sigmas = []
        num = nums[numIndx]
        print pref,num
        d = py.open(pref+str(num)+'.fits')
        v = py.open(pref[:-4]+'sum'+str(num)+'.fits') # will need to make this into a proper variance image
        slit, vslit = d[0].data, v[0].data
        vslit[vslit<=0.] = 1e9
        
        h = d[0].header
        x = np.arange(slit.shape[1])*1.
        w = h['CRVAL1']+x*h['CD1_1']
        h1,h2 = h['CRVAL1'],h['CD1_1']
        slice = np.median(vslit[5:-5,2000:7000],0)
        wslice = w[2000:7000]
        
        sky = skylines[(skylines>min(wslice))&(skylines<max(wslice))] 
        f = flux[(skylines>min(wslice))&(skylines<max(wslice))]
        sky = sky[np.argsort(f)][-30:]
        c = [slice[abs(wslice-SKY)<2.5] for SKY in sky]
        cwl = [wslice[abs(wslice-SKY)<2.5] for SKY in sky]
        
        for i in range(len(c)):
            lp,det = ngaussFit(c[i],cwl[i],sky[i],h1,h2,nsteps=250)
            sigma,offset = det
            print sigma*h2/sky[i]
            #print sigma,offset
            sigmas.append(sigma*h2/sky[i])
        sigmas = np.array(sigmas)
        np.save('/data/ljo31b/MACSJ0717/spectra/sigma'+str(num),sigmas)
        print clip(sigmas)

      
extract('/data/ljo31b/MACSJ0717/spectra/ssum',np.arange(1,31,1),plot=False)

