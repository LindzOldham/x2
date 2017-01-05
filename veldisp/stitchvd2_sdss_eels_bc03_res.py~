import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter2_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
ages = chabrier['age']
wave = chabrier['wave']
spectra = chabrier[6]

# take 2 gyr, 5 gyr, 6 gyr, 7 gyr, 9 gyr?
idx = np.where((ages==5e9)|(ages==2e9)|(ages==6e9)|(ages==7e9)|(abs(ages-9e9)<0.1e9)|(abs(ages-8e9)<0.1e9)|(abs(ages-7.5e9)<0.1e9))

dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

# science data resolution, template resolution - all esi spectra have virtually the same resolution, but may as well measure them separately!
sigtmp1=70.
sigtmp2 = light/500./2.355
print sigtmp1

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


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,srclim=6000.,lenslim=5500.,name=None):
    # Load in spectrum
    spec = py.open(dir+name+'.fits')[1].data
    sciwave = spec['loglam']
    scispec = spec['flux']
    
    varspec = spec['ivar']**-1.#-1.#-0.5
    wdisp = np.mean(spec['wdisp'][(spec['wdisp']>0)&(sciwave<np.log10(9000.))&(sciwave>np.log10(lim))])
    sigsci = wdisp*0.0001*np.log(10.)*light
    print sigsci, wdisp
    
    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(7200.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps1 = len(idx[0])
    ntemps2 = len(templates2)

    result = []
    models = []
    t1,t2 = [],[]
    
    for ii in idx[0]:
        tmpspec1 = spectra[ii]
        tmpwave1 = wave
        tmpspec1 /= tmpspec1.mean()
        twave1 = np.log10(tmpwave1)
        tmpscale1 = 7.4029335391343975e-05
        t1.append(getmodel(twave1,tmpspec1,tmpscale1,sigsci,sigtmp1))
    
    for template in templates2:
        tmpwave2,tmpspec2 = np.loadtxt(template,unpack=True)
        tmpwave2 *= 10.
        tmpspec2 /= tmpspec2.mean()
        
        twave2 = np.log10(tmpwave2)
        tmpscale2 = twave2[455]-twave2[454]
        t2.append(getmodel(twave2,tmpspec2,tmpscale2,sigsci,sigtmp2)) 

    ntemps1,ntemps2 = len(t1), len(t2)

    if fit:
        result = finddispersion(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim,bias=bias,name=name)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim,bias=bias)
        return result
    else:
        return

dir = '/data/ljo31b/EELs/sdss/'
names = py.open('/data/ljo31/Lens/LensParams/Phot_2src_lensgals_huge_new.fits')[1].data['name']
import sys
name = sys.argv[1]
print name

if name == 'J0837':
    filename='bc03_sdss_'+name
    result = run(0.4246,0.6411,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=6100,bias=1e3,srclim=6130.,lenslim=6130.,name=name)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=6100,bias=1e3,srclim=6130.,lenslim=6130.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'sdssbc03.pdf')
    pl.show()


elif name == 'J0901':
    filename = 'bc03_sdss_'+name
    #result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e2,srclim=5150.,lenslim=4750.,name=name)
    #result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e2,srclim=5150.,lenslim=4750.,name=name)

    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e3,srclim=5800.,lenslim=5000.,name=name)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e3,srclim=5800.,lenslim=5000.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0901sdssbc03.pdf')
    pl.show()


elif name == 'J0913':
    filename = 'bc03_sdss_'+name
    print name
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4500,bias=1e3,srclim=5700.,lenslim=5150.,name=name)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4500,bias=1e3,srclim=5700.,lenslim=5150.,name=name)
    
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'sdssbc03.pdf')

    pl.show()

elif name == 'J1125':
    filename = 'zv2_stitch2_sdss_'+name
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900],[7875,8240]]),nfit=12,bg='polynomial',lim=4500.,bias=1e3,srclim=6200.,lenslim=5300.,name=name)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900],[7875,8240]]),nfit=12,bg='polynomial',lim=4500.,bias=1e3,srclim=6200.,lenslim=5300.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'sdssbc03.pdf')

    pl.show()


elif name == 'J1144':
    filename = 'zv2_stitch2_sdss_'+name
    result = run(0.372,0.706,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=5000.,bias=1e2,srclim=6300.,lenslim=5050.,name=name)
    result = run(0.372,0.706,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=5000.,bias=1e2,srclim=6300.,lenslim=5050.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

    pl.show()


elif name == 'J1218':
    filename = 'zv2_stitch2_sdss_'+name
    result = run(0.3182,0.6009,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4750.,bias=1e3,srclim=5900.,lenslim=4900.,name=name)
    result = run(0.3182,0.6009,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4750.,bias=1e3,srclim=5900.,lenslim=4900.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')
    pl.show()

# now for the other half of the eels!
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]

if name == 'J1323':
    filename = 'zv_sdss_'+name
    print name, filename
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5500.,lenslim=4900.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5500.,lenslim=4900.,name=name)
    pl.title(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

elif name == 'J1347':
    filename = 'zv_sdss_'+name
    #result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=6000.,lenslim=5200.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=6000.,lenslim=5200.,name=name)
    pl.title(name)
    pl.xlim([4750,9000])
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

elif name == 'J1446':
    filename = 'zv_sdss_'+name   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5800.,lenslim=4850.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5800.,lenslim=4850.,name=name)
    pl.title(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

elif name == 'J1605':
    filename = 'zv_sdss_'+name   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5700.,lenslim=4850.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5700.,lenslim=4850.,name=name)
    pl.title(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

elif name == 'J1606':
    filename = 'zv_sdss_'+name   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=6100.,lenslim=5100.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=6100.,lenslim=5100.,name=name)
    pl.title(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')


elif name == 'J1619':
    filename = 'zv_sdss_'+name   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5900.,lenslim=5000.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e3,srclim=5900.,lenslim=5000.,name=name)
    pl.title(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

elif name == 'J2228':
    filename = 'zv_sdss_'+name   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4550.,bias=1e3,srclim=5300.,lenslim=4600.,name=name)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4550.,bias=1e3,srclim=5300.,lenslim=4600.,name=name)
    pl.title(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

pl.show()
