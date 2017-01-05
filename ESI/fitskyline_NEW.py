import pyfits as py,numpy as np,pylab as pl
from scipy import ndimage,interpolate,optimize
import special_functions as sf, indexTricks as iT
from spectra import spectools as st
import pymc
import myEmcee_blobs as myEmcee

blue = [1500,1400,1300,1200,1100,900,600,200,0,0,0]
red = [3000,3400,3700,-1,-1,-1,-1,-1,-1,-1]


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
skylines = np.log10(wl)

def ngaussFit(d,wave,mu,h1,h2,nsteps=500):
    pars = []
    logsigma = pymc.Uniform('sigma',-6,-4)
    offset = pymc.Uniform('offset',10**wave[0]-10**mu, 10**wave[-1]-10**mu,value=0)
    pars = [logsigma,offset]
    cov = np.array([0.2,0.2])
    @pymc.observed
    def logl(value=0.,pars=pars):
        lp=0
        sigma = 10**logsigma.value
        shiftmu = np.log10(10**mu-offset.value)
        #print mu, shiftmu, sigma
        model = np.ones((wave.size,2))
        model[:,1] = np.exp(-0.5*(wave-shiftmu)**2./sigma**2.)
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
    return lp,trace[a1,a2]        


def extract(pref,nums,plot=False):
    ''' here is a code to get the wavelength shift and edit the header to reflect this'''
    ospex = {} # spectrum
    ovars = {} # variance
    owave = {} # wavelength (one for each order of the echelle)
    for order in range(1,11):
        ospex[order] = []
        ovars[order] = []
    
    allsigmas = []
    for numIndx in range(len(nums)):
        num = nums[numIndx]
        print pref,num
        d = py.open('%s_%04d_bgsub.fits'%(pref,num))
        v = py.open('%s_%04d_var.fits'%(pref,num))
        
        sigmas = []
        for order in range(4,11):
            B,R = blue[order-1],red[order-1]
            slit = d[order].data.copy()
            vslit = v[order].data.copy()
            vslit[vslit<=0.] = 1e9
            vslit[np.isnan(vslit)] = 1e9
            vslit[np.isnan(slit)] = 1e9
            h = d[order].header
            x = np.arange(slit.shape[1])*1.
            #w = 10**(h['CRVAL1']+x*h['CD1_1'])
            w = h['CRVAL1']+x*h['CD1_1']
            h1,h2 = h['CRVAL1'],h['CD1_1']
            wBR = w[B:R]
            
            slice = np.median(vslit[5:-5,B:R],0)
            sky = skylines[(skylines>min(wBR))&(skylines<max(wBR))] 
            f = flux[(skylines>min(wBR))&(skylines<max(wBR))]
            if len(sky)>10:
                sky = sky[np.argsort(f)][-10:]
            c = [slice[abs(10**wBR-10**SKY)<2.5] for SKY in sky]
            cwl = [wBR[abs(10**wBR-10**SKY)<2.5] for SKY in sky]
            offsets = []
            for i in range(len(c)):
                lp,det = ngaussFit(c[i],cwl[i],sky[i],h1,h2,nsteps=250)
                sigma,offset = det
                #print 10**sigma, 10**sigma / sky[i], 10**sigma / sky[i] * np.log(10.), 10**sigma * np.log(10.)
                #print 10**sigma # this is dlambda/lambda/ln10
                #print 10**sigma * np.log(10.)
                #print 10**sigma * np.log(10.)
                sigmas.append(10**sigma * np.log(10.))#10**sigma / sky[i] * np.log(10.))#10**sigma * np.log(10.))
        sigmas = np.array(sigmas)
        #print 'clip', order, clip(sigmas)
    allsigmas.append(sigmas)
    np.save('/data/ljo31b/EELs/esi/spectra/'+str(name),allsigmas)
    print names[qq], clip(np.array(allsigmas))[0]
    resolutions.append([name[:5],clip(np.array(allsigmas))[0]])



resolutions = []
names = ['J0837+0801','J0901+2027','J0913+4237','J1125+3058','J1144+1540','J1218+5648']
refs = [[33,34,35],[36,37,38],[39,40,41],[45,46,47],[48,49,50],[51,52,53]]

for qq in range(len(names)):
    name = names[qq]
    extract('/data/ljo31b/EELs/esi/'+name[:5]+'/EEL_'+names[qq],refs[qq],plot=False)

np.save('/data/ljo31b/EELs/esi/spectra/resolutions_from_skylines_NEW',dict(resolutions))
