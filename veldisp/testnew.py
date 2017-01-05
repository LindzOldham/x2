import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter3_oneobj_vdfit import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy import sparse
from scipy.interpolate import splrep, splev
import VDfit


spec = py.open('/data/ljo31b/EELs/esi/testcode/spec-1460-53138-0505.fits')
wdisp = spec[1].data['wdisp']
sciwave = spec[1].data['loglam']
sciwave = 10**sciwave
wdispmod = splrep(sciwave,wdisp)

def sigsci(wave):
    return splev(wave,wdispmod)*0.0001*np.log(10.)*299792.458

t1 = VDfit.INDOUS(sigsci)
t2 = VDfit.PICKLES(sigsci)

def run(z,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,lenslim=5500.):
    # Load in spectrum
    spec = py.open('/data/ljo31b/EELs/esi/testcode/spec-1460-53138-0505.fits')[1].data
    sciwave,scispec,varspec = spec['loglam'],spec['flux'],spec['ivar']**-1

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(7500.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2
    
    
    if fit:
        result = finddispersion(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),z,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,lenslim=lenslim,bias=bias)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),z,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,lenslim=lenslim,bias=bias)
        return result
    else:
        return


name = 'testcode_stitch2_neu_bc03'


result = run(0.0804,fit=True,read=False,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=4,bg='polynomial',lim=3500.,bias=1e2,lenslim=3800.)
result = run(0.0804,fit=False,read=True,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=4,bg='polynomial',lim=3500.,bias=1e2,lenslim=3800.)

lp,trace,dic,_=result
trace=trace[:,lp[200]>-1200]

pl.title(name[:5])
pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/'+name[:5]+'.pdf')
pl.xlim([4000,10000])
pl.show()



result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+name)
lp,trace,dic,_ = result
velL,sigL= np.median(trace[100:,:].reshape(((trace.shape[0]-100)*trace.shape[1],trace.shape[2])),axis=0)
print velL, sigL

# uncertainties
f = trace[100:,:].reshape(((trace.shape[0]-100)*trace.shape[1],trace.shape[2]))
velLlo, sigLlo = np.percentile(f,16,axis=0)
velLhi,sigLhi = np.percentile(f,84,axis=0)

print 'lens velocity', '%.2f'%velL, '\pm', '%.2f'%(velL-velLlo)
print 'lens dispersion', '%.2f'%sigL, '\pm', '%.2f'%(sigL-sigLlo)

pl.figure()
pl.subplot(221)
key = 'lens velocity'
pl.hist(dic[key][200:].ravel(),bins=np.linspace(-250,400,30),histtype='stepfilled')
pl.title(key)
pl.subplot(222)
key = 'lens dispersion'
pl.hist(dic[key][200:].ravel(),30,histtype='stepfilled')
pl.title(key)
pl.axvline(102.93,color='k')
pl.show()
