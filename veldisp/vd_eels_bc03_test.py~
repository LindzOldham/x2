import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from fitter_bc03 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
import glob

chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
ages = chabrier['age']
wave = chabrier['wave']
spectra = chabrier[6]

# take 2 gyr, 5 gyr, 6 gyr, 7 gyr, 9 gyr?
idx = np.where((ages==5e9)|(ages==2e9)|(ages==6e9)|(ages==7e9)|(abs(ages-9e9)<0.1e9)|(abs(ages-8e9)<0.1e9)|(abs(ages-7.5e9)<0.1e9))
print idx, len(idx[0])

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

# science data resolution, template resolution
#sigtmp =  70. # apparently

ntemps = len(idx[0])

def getmodel(twave,tspec,tscale,sigsci,sigtmp,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        dispkern = (disp**2.+sigsci**2.-sigtmp**2.)**0.5
        #print disp,sigsci,sigtmp, disp**2.+sigsci**2.-sigtmp**2.
        if np.isnan(dispkern)==True:
            dispkern = 5. 
        kernel = dispkern/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)


def clip(arr,nsig=2.5):
    a = np.sort(arr.ravel())
    a = a[a.size*0.001:a.size*0.999]
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s

def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=6000.,nfit=6.,bg='polynomial',bias=1e8):
    # Load in spectrum
    spec = py.open(dir+name+'.fits')[1].data
    sciwave = spec['loglam']
    scispec = spec['flux']
    #scispec /= np.mean(scispec)
    varspec = spec['ivar']**-1
    wdisp = np.mean(spec['wdisp'])
    sigsci = wdisp*0.0001*np.log(10.)*light
    print sigsci

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = sciwave.size
    end = np.where(sciwave>np.log10(9000.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]
    
    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    t=[]
    sigtmp = 3./np.mean(sciwave/(1.+zl))/2.355 * 3e5
    print 'sigtmp', sigtmp, '!!'

    for ii in idx[0]:
        tmpspec = spectra[ii]
        tmpwave = wave
        tmpspec /= tmpspec.mean()
        twave = np.log10(tmpwave)
        tmpscale = 7.4029335391343975e-05#twave1[2000]-twave1[1999]#7.4029335391343975e-05#twave1[2000]-twave1[1999]
        t.append(getmodel(twave,tmpspec,tmpscale,sigsci,sigtmp))

    if fit:
        result = finddispersion(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,bias=bias)
        return result
    elif read:
        result = readresults(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,bias=bias)
        return result
    else:
        return

import sys
name = sys.argv[1]
print name

dir = '/data/ljo31b/EELs/sdss/'
names = py.open('/data/ljo31/Lens/LensParams/Phot_2src_lensgals_huge_new.fits')[1].data['name']
filename='bc03_sdss_'+name

if name == 'J0837':
    result = run(0.4256,0.6411,fit=True,read=False,File = filename+'_2_7000to9000',mask=np.array([[6300,6320],[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=7000,bias=1e2)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename+'_2_7000to9000',mask=np.array([[6300,6320],[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=7000,bias=1e2)
    pl.suptitle(name)
    pl.xlim([6000,8000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0837_sdss_bc03_seg2_7000to9000.png')


elif name == 'J0901':
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e1)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e1)
    pl.suptitle(name[:5])
    pl.xlim([6000,9000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'_sdss_bc03_seg..png')

elif name == 'J0913':
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5700,bias=1e1)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5700,bias=1e1)
    pl.suptitle(name)
    pl.xlim([6000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'_sdss_bc03_seg..png')

elif name == 'J1125':
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6200,bias=1e1)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6200,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'_sdss_bc03_seg..png')


elif name == 'J1144':
    result = run(0.372,0.706,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6300,bias=1e1)
    result = run(0.372,0.706,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6300,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'.png')

elif name == 'J1218':
    result = run(0.3182,0.6009,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5900.,bias=1e1)
    result = run(0.3182,0.6009,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5900.,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'.png')

pl.show()
