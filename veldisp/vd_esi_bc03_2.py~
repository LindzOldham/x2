import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from fitter import finddispersion,readresults
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

def getmodel4(twave,tspec,tscale,smin=5.,smax=501):
    match = tspec.copy()
    kernel = (sigsci**2.+50.**2-sigtmp1**2)**0.5/(299792.*ln10*tscale)
    match = ndimage.gaussian_filter1d(tspec.copy(),kernel)
    disps = np.arange(smin,smax,VGRID)
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        disp = (disp**2.-50.**2.)**0.5
        if np.isnan(disp)==True:
            disp = 5.
        kernel = disp/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)


def getmodel(twave,tspec,tscale,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        dispkern = (disp**2.+sigsci**2.-sigtmp1**2.)**0.5
        if np.isnan(dispkern)==True:
            dispkern = 5.
        kernel = dispkern/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)

def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=6000.,nfit=6.,bg='polynomial',bias=15.):
    # Load in spectrum
    scispec = py.open('/data/ljo31b/EELs/esi/spectra/'+name[:5]+'_spec.fits')[0].data
    varspec = py.open('/data/ljo31b/EELs/esi/spectra/'+name[:5]+'_var.fits')[0].data
    sciwave = py.open('/data/ljo31b/EELs/esi/spectra/'+name[:5]+'_wl.fits')[0].data

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(9500.))[0][0]
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

        # do we need to smooth these? Presumably not, because our resolution is better
        twave = np.log10(tmpwave)
        tmpscale = twave[1]-twave[0]
        t.append(getmodel(twave,tmpspec,tmpscale)) 
    if fit:
        result = finddispersion(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,bias=bias)
        return result
    elif read:
        result = readresults(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,bias=bias)
        return result
    else:
        return

if name == 'J0837':
    result = run(0.4256,0.6411,fit=True,read=False,File = filename+'_2_7000to9000',mask=np.array([[6300,6320],[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=7000,bias=1e2)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename+'_2_7000to9000',mask=np.array([[6300,6320],[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=7000,bias=1e2)
    pl.suptitle(name)
    pl.xlim([6000,8000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0837_esi_bc03_seg2_7000to9000.png')


elif name == 'J0901':
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e1)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e1)
    pl.suptitle(name[:5])
    pl.xlim([6000,9000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'_esi_bc03_seg..png')

elif name == 'J0913':
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5700,bias=1e1)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=5700,bias=1e1)
    pl.suptitle(name)
    pl.xlim([6000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'_esi_bc03_seg..png')

elif name == 'J1125':
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6200,bias=1e1)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[5520,5620],[7580,7700],[6860,6900]]),nfit=9,bg='polynomial',lim=6200,bias=1e1)
    pl.title(name)
    pl.xlim([4000,10000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'_esi_bc03_seg..png')


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
