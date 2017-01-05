import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter2_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy import sparse
from scipy.interpolate import splrep, splev

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

def get_bc03_model(sigsci,name):
    pixscale = 1.
    tempSigma = light*3./wave/2.355
    X = np.arange(wave.size)
    disps = np.arange(130.,500.,1.)
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
    np.save('/data/ljo31b/EELs/esi/BC03models_EELs_'+name,model)

dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]

sigtmp2 = light/500./2.355

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


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,srclim=6000.,lenslim=5500.,name=None,make=False,maxlim=8500.):
    # Load in spectrum
    spec = py.open(dir+name+'.fits')[1].data
    sciwave = spec['loglam']
    scispec = spec['flux']
    
    varspec = spec['ivar']**-1.
    wdisp = spec['wdisp']
    model = splrep(sciwave,wdisp,k=1)
    sigsci = splev(np.log10(wave),model)*0.0001*np.log(10.)*light
    #sigsci1 = np.mean(wdisp[(sciwave<np.log10(maxlim))&(sciwave>np.log10(lim))])*0.0001*np.log(10.)*light
    #print sigsci1

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(maxlim))[0][0]
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

    t1,t2 = [],[]
    
    if make is True:
        get_bc03_model(sigsci,name)
        print 'made splines'

    t1 = np.load('/data/ljo31b/EELs/esi/BC03models_EELs_'+name+'.npy')    
    tmpwave1 = wave.copy()
    
    sigsci2 = np.mean(wdisp[(sciwave<srclim)&(sciwave>lim)])*0.0001*np.log(10.)*light
    print sigsci2

    for template in templates2:
        tmpwave2,tmpspec2 = np.loadtxt(template,unpack=True)
        tmpwave2 *= 10.
        tmpspec2 /= tmpspec2.mean()
        
        twave2 = np.log10(tmpwave2)
        tmpscale2 = twave2[455]-twave2[454]
        t2.append(getmodel(twave2,tmpspec2,tmpscale2,sigsci2,sigtmp2)) 

    ntemps1,ntemps2 = len(t1), len(t2)

    if fit:
        result = finddispersion(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim,bias=bias,name=name)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim,bias=bias)
        return result
    else:
        return

dir = '/data/ljo31b/EELs/sdss/'
names = py.open('/data/ljo31/Lens/LensParams/Phot_2src_lensgals_huge_new.fits')[1].data['name']
import sys
name = sys.argv[1]
print name

if name == 'J0837':
    filename=name+'_sdss_bc03'
    result = run(0.4246,0.6411,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900,bias=1e3,srclim=5300.,lenslim=5000.,name=name,make=True,maxlim=8500.)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900,bias=1e3,srclim=5300.,lenslim=5000.,name=name,maxlim=8500.)
    


elif name == 'J0901':
    filename = name+'_sdss_bc03'
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900,bias=1e2,srclim=5300.,lenslim=4950.,name=name,make=True,maxlim=8500.)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900,bias=1e2,srclim=5300.,lenslim=4950.,name=name,maxlim=8500.)



elif name == 'J0913':
    filename = name+'_sdss_bc03_x'
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e3,srclim=5000.,lenslim=4600.,name=name,make=True,maxlim=8500.)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e3,srclim=5000.,lenslim=4600.,name=name,maxlim=8500.)
    

elif name == 'J1125':
    filename = name+'_sdss_bc03_x'
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900],[7875,8240]]),nfit=12,bg='polynomial',lim=4000.,bias=1e3,srclim=5500.,lenslim=5300.,name=name,make=True,maxlim=8500)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900],[7875,8240]]),nfit=12,bg='polynomial',lim=4000.,bias=1e3,srclim=5500.,lenslim=5300.,name=name,maxlim=8500.)
    


elif name == 'J1144':
    filename = name+'_sdss_bc03'
    result = run(0.372,0.706,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=5000.,bias=1e2,srclim=5500.,lenslim=5050.,name=name,make=True,maxlim=8500)
    result = run(0.372,0.706,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=5000.,bias=1e2,srclim=5500.,lenslim=5050.,name=name,maxlim=8500.)
    


elif name == 'J1218':
    filename = name+'_sdss_bc03'
    result = run(0.3182,0.6009,fit=True,read=False,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4800.,bias=1e3,srclim=5200.,lenslim=4900.,name=name,make=True,maxlim=8500)
    result = run(0.3182,0.6009,fit=False,read=True,File = filename,mask=np.array([[5570,5585],[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4800.,bias=1e3,srclim=5200.,lenslim=4900.,name=name,maxlim=8500.)
    

# now for the other half of the eels!
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]

if name == 'J1323':
    filename = name+'_sdss_bc03'
    print name, filename
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=4920.,lenslim=4920.,name=name,make=True,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=4920.,lenslim=4920.,name=name,maxlim=8500.)
    

elif name == 'J1347':
    filename = name+'_sdss_bc03'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5250.,lenslim=4920.,name=name,make=False,maxlim=8000.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5250.,lenslim=4920.,name=name,maxlim=8000.)
    
elif name == 'J1446':
    filename = name+'_sdss_bc03' 
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5100.,lenslim=4920.,name=name,make=True,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5100.,lenslim=4920.,name=name,maxlim=8500.)
   
elif name == 'J1605':
    filename = name+'_sdss_bc03' 
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5000.,lenslim=4920.,name=name,make=True,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5000.,lenslim=4920.,name=name,maxlim=8500.)
   

elif name == 'J1606':
    filename = name+'_sdss_bc03'   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5350.,lenslim=4920.,name=name,make=True,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5350.,lenslim=4920.,name=name,maxlim=8500.)
    

elif name == 'J1619':
    filename = name+'_sdss_bc03'   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5250.,lenslim=4920.,name=name,make=True,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4900.,bias=1e3,srclim=5250.,lenslim=4920.,name=name,maxlim=8500.)
    

elif name == 'J2228':
    filename = name+'_sdss_bc03'   
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4700.,bias=1e3,srclim=4720.,lenslim=4720.,name=name,make=True,maxlim=8000.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=10,bg='polynomial',lim=4700.,bias=1e3,srclim=4720.,lenslim=4720.,name=name,make=True,maxlim=8000.)
    

lp,trace,dic,_=result
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/'+filename+'_chain.dat',trace[200:].reshape((trace[200:].shape[0]*trace.shape[1],trace.shape[-1])),header=' lens velocity -- lens dispersion -- source velocity -- source dispersion \n -500,500, 0,500, -500,500, 0,500')

pl.suptitle(name)
pl.savefig('/data/ljo31b/EELs/esi/kinematics/inference/plots/'+filename+'.pdf')
pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+filename+'.pdf')
pl.figure()
pl.plot(lp)
pl.show()

from tools import iCornerPlotter as i
i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/'+filename+'_chain.dat,blue'])
pl.show()
