import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter2_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy import sparse

dir = '/data/ljo31b/EELs/esi/INDOUS/' 
templates1 = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates1)):
    templates1[i] = dir+templates1[i]

spex = [py.open(template)[0].data.astype(np.float64) for template in templates1]
spex = [spec/spec.mean() for spec in spex]
wave = T.wavelength(templates1[0],0)  # same for all of them

light = 299792.458
ln10 = np.log(10.)

def get_bc03_model(sigsci,name):
    logpixscale = np.log10(wave[1])-np.log10(wave[0])
    tempSigma = light*1.2/wave/2.355
    X = np.arange(wave.size)
    disps = np.arange(130.,500.,1.)
    ogrids = [np.empty((disps.size,wave.size)) for s in spex]
    for i in range(disps.size):
        kernel = (disps[i]**2. + sigsci**2. - tempSigma**2.)**0.5/light
        kpix = kernel/(ln10*logpixscale)
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
    np.save('/data/ljo31b/EELs/esi/INDOUSmodels_EELs_'+name,model)


dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]

sigtmp2 = light/500.

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


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,srclim=6000.,lenslim=5500.,name=None,make=False):
    # Load in spectrum
    spec = py.open(dir+name+'.fits')[1].data
    sciwave = spec['loglam']
    scispec = spec['flux']
    
    varspec = spec['ivar']**-1.#-1.#-0.5
    wdisp = np.mean(spec['wdisp'][(spec['wdisp']>0)&(sciwave<np.log10(9000.))&(sciwave>np.log10(lim))])
    sigsci = wdisp*0.0001*np.log(10.)*light

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(8500.))[0][0]
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
    
    if make is True:
        get_bc03_model(sigsci,name)
        print 'made splines'

    t1 = np.load('/data/ljo31b/EELs/esi/INDOUSmodels_EELs_'+name+'.npy')    
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
    filename='zv_stitch2_sdss_'+name
    result = run(0.4246,0.6411,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=13,bg='polynomial',lim=5000,bias=1e2,srclim=6000.,lenslim=5500.,name=name,make=True)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=13,bg='polynomial',lim=5000,bias=1e2,srclim=6000.,lenslim=5500.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')
    pl.show()


elif name == 'J0901':
    filename = 'zv2_stitch2_sdss_'+name
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=6000,bias=1e4,srclim=6200.,lenslim=6100.,name=name)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=6000,bias=1e4,srclim=6200.,lenslim=6100.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0901sdss2.pdf')



elif name == 'J0913':
    filename = 'zv2_stitch2_sdss_'+name
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4500,bias=1e2,srclim=5700.,lenslim=5150.,name=name)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4500,bias=1e2,srclim=5700.,lenslim=5150.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

    pl.show()

elif name == 'J1125':
    filename = 'zv2_stitch2_sdss_'+name
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900],[7875,8240]]),nfit=12,bg='polynomial',lim=4500.,bias=1e2,srclim=6200.,lenslim=5300.,name=name)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900],[7875,8240]]),nfit=12,bg='polynomial',lim=4500.,bias=1e2,srclim=6200.,lenslim=5300.,name=name)
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+name+'_full_zv.pdf')

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

