import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter3_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy import sparse

chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
ages = chabrier['age']
wave = chabrier['wave']
spectra = chabrier[6]
c = (wave>3000.)&(wave<8000.)
wave=wave[c]
# take 2 gyr, 5 gyr, 6 gyr, 7 gyr, 9 gyr?
idx = np.where((ages==5e9)|(ages==2e9)|(ages==6e9)|(ages==7e9)|(abs(ages-9e9)<0.1e9)|(abs(ages-8e9)<0.1e9)|(abs(ages-7.5e9)<0.1e9))
spex = [spectra[ii][c]/spectra[ii][c].mean() for ii in idx[0]]

light = 299792.458
ln10 = np.log(10.)

#sigsci = 21.
sourcespec = py.open('/data/ljo31b/EELs/esi/testcode/spec-1653-53534-0550.fits')
lenspec = py.open('/data/ljo31b/EELs/esi/testcode/spec-2157-54242-0324.fits')

wdisp1, wdisp2 = lenspec[1].data['wdisp'].mean(), sourcespec[1].data['wdisp'].mean()
sig = np.mean((wdisp1,wdisp2))
sig *= 0.0001*np.log(10.) # 0.0001 = cd1_1, dispersion of each pixel; fixed at 69 km/s = 0.0001 * ln(10) * c
sigsci = sig * light
print sigsci

pixscale = 1.
tempSigma = light*3./wave/2.355
X = np.arange(wave.size)
disps = np.arange(130.,400.,1.)
ogrids = [np.empty((disps.size,wave.size)) for s in spex]
for i in range(disps.size):
    kernel = wave*(disps[i]**2. + sigsci**2. - tempSigma**2.)**0.5/light
    kpix = kernel/pixscale
    kpix2 = kpix**2.
    pmax = int(kpix.max()*4+1)
    rcol = np.linspace(-pmax,pmax,2*pmax+1).repeat(X.size).reshape((2*pmax+1,X.size))
    col = rcol+X
    row = X.repeat(2*pmax+1).reshape((X.size,2*pmax+1)).T
    c = (col>=0)&(col<wave.size)&(abs(rcol)<4*kpix[row])
    col = col[c]
    row = row[c]
    rcol = rcol[c]
    kpix2 = kpix2[row]
    wht = np.exp(-0.5*rcol**2/kpix2)/(2*np.pi*kpix2)**0.5
    M = sparse.coo_matrix((wht,(row,col)),shape=(X.size,X.size))

    for j in range(len(spex)):
        s = spex[j].copy()
        s = M*s
        ogrids[j][i] = s

X = disps.tolist()
tx = np.array([X[0]]+X+[X[-1]])
Y = np.log10(wave).tolist()
ty = np.array([Y[0]]+Y+[Y[-1]])
model = [(tx,ty,ogrid.flatten(),1,1) for ogrid in ogrids]
np.save('/data/ljo31b/EELs/esi/BC03models',model)


'''
make these templates here, ok!
wave[400] = 3375
wave[5000] = 7975
and in between these regions (plus a bit further on either side), they are sampled at a constant dlambda = 1 AA. At the edges, they are sparser, but this doesn't need to worry us.


'''
dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']
sigtmp2 = light/500./2.355

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]


def getmodel(twave,tspec,tscale,sigsci,sigtmp,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,1.)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        dispkern = (disp**2.+sigsci**2.-sigtmp**2.)**0.5
        if np.isnan(dispkern)==True:
            dispkern = 5. 
        kernel = dispkern/(light*ln10*tscale)
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
    end = np.where(sciwave>np.log10(6500.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2
   
    t1,t2 = [],[]
    t1 = np.load('/data/ljo31b/EELs/esi/BC03models.npy')    
    tmpwave1 = wave.copy()

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


name = 'testcode_stitch2_neu_bc03'
sourcespec = py.open('/data/ljo31b/EELs/esi/testcode/spec-1653-53534-0550.fits')
lenspec = py.open('/data/ljo31b/EELs/esi/testcode/spec-2157-54242-0324.fits')

wdisp1, wdisp2 = lenspec[1].data['wdisp'].mean(), sourcespec[1].data['wdisp'].mean()
sig = np.mean((wdisp1,wdisp2))
sig *= 0.0001*np.log(10.) # 0.0001 = cd1_1, dispersion of each pixel; fixed at 69 km/s = 0.0001 * ln(10) * c
sigsci = sig * light
print sigsci
#result = run(0.111974,0.311078,fit=True,read=False,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4000.,bias=1e10,srclim=4800.,lenslim=4100.)
#result = run(0.111974,0.311078,fit=False,read=True,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4000.,bias=1e10,srclim=4800.,lenslim=4100.)
result = run(0.111974,0.311078,fit=True,read=False,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4000.,bias=1e10,srclim=4800.,lenslim=4100.)
result = run(0.111974,0.311078,fit=False,read=True,File = name,mask=np.array([[5560,5620],[6300,6320],[7225,7280],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4000.,bias=1e10,srclim=4800.,lenslim=4100.)



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
