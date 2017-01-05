import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter3_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy.interpolate import splrep, splev
from scipy import sparse

dir = '/data/ljo31b/EELs/esi/INDOUS/' 
templates1 = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates1)):
    templates1[i] = dir+templates1[i]

spex = [py.open(template)[0].data.astype(np.float64) for template in templates1]
spex = [spec/spec.mean() for spec in spex]
wave = T.wavelength(templates1[0],0)  # same for all of them

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]

import sys
name,wid,typ = sys.argv[1],sys.argv[2],sys.argv[3]
triangle=False

light = 299792.458
ln10 = np.log(10.)
may2014 = ['J1323','J1347','J1446','J1605','J1606','J1619','J2228']
if name in may2014:
    sigsci = 20.26
else:
    sigsci = 20.40

logpixscale = np.log10(wave[1])-np.log10(wave[0])
tempSigma = light*1.2/wave/2.355
X = np.arange(wave.size)
disps = np.arange(80.,500.,1.)
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
np.save('/data/ljo31b/EELs/esi/INDOUSmodels_EELs_may2014',model)

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


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,srclim=6000.,lenslim=5500.,wid=1.,typ=None,maxlim=8500.):
    # Load in spectrum
    if typ=='lens':
        print 'lens'
        try:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_spec_lens.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_var_lens.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_wl_lens.fits')[0].data
        except:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_spec_lens.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_var_lens.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_wl_lens.fits')[0].data
    else:
        print 'source'
        try:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_spec_source.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_var_source.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_wl_source.fits')[0].data
        except:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_spec_sourceb.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_var_sourceb.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_wl_sourceb.fits')[0].data
    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(maxlim))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]
    
    # convert esi air wavelengths to vacuum wavelengths
    # sdss website, cf, morton 1991
    vac = np.linspace(3800,11000,(11000-3800+1)*5)
    air = vac / (1. + 2.735182e-4 + (131.4182/vac**2.) + (2.76249e8/vac**8))
    model = splrep(air,vac)
    sciwave = splev(sciwave,model)
    # see what difference this makes

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps1 = len(templates1)
    ntemps2 = len(templates2)

    result = []
    models = []
    t1,t2 = [],[]
    
    if name in may2014:
        t1 = np.load('/data/ljo31b/EELs/esi/INDOUSmodels_EELs_may2014.npy')    
    else:
        t1 = np.load('/data/ljo31b/EELs/esi/INDOUSmodels_EELs.npy')
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


if name=='J0837':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(0.4246,0.6411,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5295,5325],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=5000,bias=1e10,srclim=6000.,lenslim=5500.,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5295,5325],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=5000,bias=1e10,srclim=6000.,lenslim=5500.,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J0901':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,srclim=5800.,lenslim=5000.,typ=typ,wid=wid,maxlim=8000.)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,srclim=5800.,lenslim=5000.,typ=typ,wid=wid,maxlim=8000.)
    

elif name == 'J0913':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4500,bias=1e8,srclim=5700.,lenslim=5150.,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4500,bias=1e8,srclim=5700.,lenslim=5150.,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J1125':
    filename = name+'_'+wid+'_'+typ +'_esi_indous'
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4500.,bias=1e8,srclim=5500.,lenslim=5300.,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4500.,bias=1e8,srclim=5500.,lenslim=5300.,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J1144':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(0.372,0.706,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=5000.,bias=1e8,srclim=6300.,lenslim=5050.,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.372,0.706,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=5000.,bias=1e8,srclim=6300.,lenslim=5050.,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J1218':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(0.3182,0.6009,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4800.,bias=1e8,srclim=5900.,lenslim=4900.,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.3182,0.6009,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4800.,bias=1e8,srclim=5900.,lenslim=4900.,typ=typ,wid=wid,maxlim=8500.)
    
# may 2014 dataset

elif name == 'J1323':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5400.,lenslim=4920.,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5400.,lenslim=4920.,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1347':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e18,srclim=6000.,lenslim=5100.,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e18,srclim=6000.,lenslim=5100.,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1446':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5800.,lenslim=4920.,wid=wid,typ=typ,maxlim=8000.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5800.,lenslim=4920.,wid=wid,typ=typ,maxlim=8000.)


elif name == 'J1605':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5700.,lenslim=4920.,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5700.,lenslim=4920.,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1606':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=6100.,lenslim=5100.,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=6100.,lenslim=5100.,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1619':
    filename = name+'_'+wid+'_'+typ+'_esi_bc03'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5900.,lenslim=5050.,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,srclim=5900.,lenslim=5050.,wid=wid,typ=typ,maxlim=8500.)


elif name == 'J2228':
    filename = name+'_'+wid+'_'+typ+'_esi_indous'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4700.,bias=1e8,srclim=5300.,lenslim=4710.,wid=wid,typ=typ,maxlim=8000.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4700.,bias=1e8,srclim=5300.,lenslim=4710.,wid=wid,typ=typ,maxlim=8000.)
    


lp,trace,dic,_=result
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/'+filename+'_chain.dat',trace[200:].reshape((trace[200:].shape[0]*trace.shape[1],trace.shape[-1])),header=' lens velocity -- lens dispersion -- source velocity -- source dispersion \n -500,500, 0,500, -500,500, 0,500')

pl.suptitle(name)
pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/'+filename+'.pdf')
pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+filename+'.png')
#pl.show()

from tools import iCornerPlotter as i
i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/'+filename+'_chain.dat,blue'])
#pl.show()
pl.close('all')
