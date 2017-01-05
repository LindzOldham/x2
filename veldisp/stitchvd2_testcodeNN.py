import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter3_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g


dir = '/data/ljo31b/EELs/esi/INDOUS/' 
templates1 = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates1)):
    templates1[i] = dir+templates1[i]

dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

# science data resolution, template resolution - all esi spectra have virtually the same resolution, but may as well measure them separately!
sigtmp1 = np.mean((1./5500,1./3900))*1.2 * light / 2.355
sigtmp2 = light/500./2.355

def getmodel(twave,tspec,tscale,sigsci,sigtmp,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        dispkern = (disp**2.+sigsci**2.-sigtmp**2.)**0.5
        if np.isnan(dispkern)==True:
            dispkern = 5. 
        kernel = dispkern/(light*ln10*tscale)
        print kernel
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,srclim=6000.,lenslim=5500.):
    # Load in spectrum
    sciwave,scispec,varspec = np.load('/data/ljo31b/EELs/esi/testcode/summedspec_neu_neu.npy')

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(6500.))[0][0]#scispec.size
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2
   
    # prepare the templates
    ntemps1 = len(templates1)
    ntemps2 = len(templates2)

    result = []
    models = []
    t1,t2 = [],[]
    
    for template in templates1:
        file = py.open(template)
        tmpspec1 = file[0].data.astype(np.float64)
        tmpwave1 = T.wavelength(template,0)
        tmpspec1 /= tmpspec1.mean()

        twave1 = np.log10(tmpwave1)
        tmpscale1 = twave1[1]-twave1[0]
        t1.append(getmodel(twave1,tmpspec1,tmpscale1,sigsci,sigtmp1)) 
    
    for template in templates2:
        tmpwave2,tmpspec2 = np.loadtxt(template,unpack=True)
        tmpwave2 *= 10.
        tmpspec2 /= tmpspec2.mean()

        twave2 = np.log10(tmpwave2)
        tmpscale2 = twave2[1]-twave2[0]
        t2.append(getmodel(twave2,tmpspec2,tmpscale2,sigsci,sigtmp2)) 

    ntemps1,ntemps2 = len(t1), len(t2)

    if fit:
        result = finddispersion(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim)
        return result
    else:
        return




name = 'testcode_stitch2_neu'
sourcespec = py.open('/data/ljo31b/EELs/esi/testcode/spec-1653-53534-0550.fits')
lenspec = py.open('/data/ljo31b/EELs/esi/testcode/spec-2157-54242-0324.fits')

wdisp1, wdisp2 = lenspec[1].data['wdisp'].mean(), sourcespec[1].data['wdisp'].mean()
sig = np.mean((wdisp1,wdisp2))
sig *= 0.0001*np.log(10.) # 0.0001 = cd1_1, dispersion of each pixel; fixed at 69 km/s = 0.0001 * ln(10) * c
sigsci = sig * light
print sigsci
#result = run(0.111974,0.311078,fit=True,read=False,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4500.,bias=1e10,srclim=4800.,lenslim=4600.)
#result = run(0.111974,0.311078,fit=False,read=True,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4500.,bias=1e10,srclim=4800.,lenslim=4600.)

result = run(0.111974,0.311078,fit=True,read=False,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4000.,bias=1e10,srclim=4800.,lenslim=4100.)
result = run(0.111974,0.311078,fit=False,read=True,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4000.,bias=1e10,srclim=4800.,lenslim=4100.)
#pl.title(name[:5])

pl.title(name[:5])
pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/'+name[:5]+'.pdf')
pl.xlim([4000,10000])
pl.show()



result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/final/'+name)
lp,trace,dic,_ = result
velL,sigL,velS,sigS = np.median(trace[100:,:].reshape(((trace.shape[0]-100)*trace.shape[1],trace.shape[2])),axis=0)
print velL, sigL, velS, sigS

# uncertainties
f = trace[100:,:].reshape(((trace.shape[0]-100)*trace.shape[1],trace.shape[2]))
velLlo, sigLlo,velSlo,sigSlo = np.percentile(f,16,axis=0)
velLhi,sigLhi,velShi,sigShi = np.percentile(f,84,axis=0)

print 'lens velocity', '%.2f'%velL, '\pm', '%.2f'%(velL-velLlo)
print 'lens dispersion', '%.2f'%sigL, '\pm', '%.2f'%(sigL-sigLlo)
print 'source velocity', '%.2f'%velS, '\pm', '%.2f'%(velS-velSlo)
print 'source dispersion', '%.2f'%sigS, '\pm', '%.2f'%(sigS-sigSlo)

pl.figure()
pl.subplot(221)
key = 'lens velocity'
pl.hist(dic[key][100:].ravel(),30,histtype='stepfilled')
pl.title(key)
pl.subplot(222)
key = 'lens dispersion'
pl.hist(dic[key][100:].ravel(),30,histtype='stepfilled')
pl.title(key)
pl.axvline(229.57,color='k')
pl.subplot(223)
key = 'source velocity'
pl.hist(dic[key][100:].ravel(),30,histtype='stepfilled')
pl.title(key)
pl.subplot(224)
key = 'source dispersion'
pl.hist(dic[key][100:].ravel(),30,histtype='stepfilled')
pl.title(key)
pl.axvline(272.33,color='k')
pl.show()
