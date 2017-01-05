import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from fittertest import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

dir = '/data/ljo31b/EELs/esi/INDOUS/'
templates = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates)):
    templates[i] = dir+templates[i]

twave  = np.log10(T.wavelength(templates[0],0))
tscale = twave[1]-twave[0]

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

# science data resolution, template resolution
sigtmp1 =  0.44/7000. * light
sigtmp2 = light/500.

def clip(arr,nsig=2.5):
    a = np.sort(arr.ravel())
    a = a[a.size*0.001:a.size*0.999]
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s

def getmodel(twave,tspec,tscale,sigsci,smin=5.,smax=601):#? due to large dispersions, seems necessary...
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    kernel = (sigsci**2-sigtmp1**2)**0.5/(299792.*ln10*tscale)
    match = ndimage.gaussian_filter1d(tspec.copy(),kernel)
    for i in range(disps.size):
        disp = disps[i]
        kernel = disp/(299792.*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)

def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=6000.,nfit=6.,bg='polynomial',bias=1e8):
    # Load in spectrum
    spec = py.open(dir+name+'.fits')[1].data
    sciwave = spec['loglam']
    scispec = spec['flux']
    scispec /= np.mean(scispec)
    varspec = spec['ivar']**-0.5#-1#-2
    wdisp = np.mean(spec['wdisp'])
    sigsci = wdisp*0.0001*np.log(10.)*light
    print sigsci

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = sciwave.size
    end = np.where(sciwave>np.log10(8500.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]
    
    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps = len(templates)
    ntemp = 1
    result = []
    models = []
    t = []

    tmin = sciwave.min()
    tmax = sciwave.max()

    for template in templates:
        # Load the template
        file = py.open(template)
        tmpspec = file[0].data.astype(np.float64)
        tmpwave = T.wavelength(template,0)
        tmpspec /= tmpspec.mean()
        print min(tmpwave)
        # do we need to smooth these? Presumably not, because our resolution is better
        twave = np.log10(tmpwave)
        tmpscale = twave[1]-twave[0]
        t.append(getmodel(twave,tmpspec,tmpscale,sigsci)) 
    np.save('/data/ljo31b/EELs/esi/testcode/testtemplates2',t)
    np.save('/data/ljo31b/EELs/esi/testcode/testtwave2',tmpwave)
        #pl.figure()
        #pl.plot(tmpwave,tmpspec)
        #pl.show()
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
filename='sdss_'+name

if name == 'J0837':
    result = run(0.4256,0.6411,fit=True,read=False,File = filename,mask=np.array([[6300,6320],[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=6000,bias=1e1)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[6300,6320],[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=6000,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'.png')


elif name == 'J0901':
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e1)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e1)
    pl.title(name[:5])
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'.png')

elif name == 'J0913':
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5700,bias=1e1)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5700,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'.png')

elif name == 'J1125':
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6200,bias=1e1)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6200,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'.png')


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

#pl.show()
