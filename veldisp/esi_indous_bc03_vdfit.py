import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter_esi_vdfit import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy.interpolate import splrep, splev
from scipy import sparse
import VDfit

import sys
name,wid,typ,nthreads = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
triangle=False

may2014 = ['J1323','J1347','J1446','J1605','J1606','J1619','J2228']
if name in may2014:
    sigsci = lambda wave: 20.26
else:
    sigsci = lambda wave: 20.40

t1 = VDfit.INDOUS(sigsci)
t2 = VDfit.BC03(sigsci)

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]


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

    lenslim = 3650.*(1.+zl)+40.
    srclim = 3650.*(1.+zs)+40.
    print lenslim, srclim
    if lenslim<=lim:
        lenslim = lim+10.
    if srclim<=lim:
        srclim = lim+10.
    #srclim,lenslim = 6000.,5500.
    print 'lims: ', lim, lenslim, srclim

    if fit:
        print 'running on ', nthreads, ' threads'
        result = finddispersion(scispec,varspec,t1,t2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim,nthr=nthreads)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,srclim=srclim,lenslim=lenslim)
        return result
    else:
        return


if name=='J0837':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit_old_two'
    result = run(0.4246,0.6411,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5295,5325],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=5000,bias=1e10,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5295,5325],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=5000,bias=1e10,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J0901':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,typ=typ,wid=wid,maxlim=8000.)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,typ=typ,wid=wid,maxlim=8000.)
    

elif name == 'J0913':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J1125':
    filename = name+'_'+wid+'_'+typ +'_esi_indous_vdfit'
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J1144':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(0.372,0.706,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.372,0.706,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    

elif name == 'J1218':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(0.3182,0.6009,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    result = run(0.3182,0.6009,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,typ=typ,wid=wid,maxlim=8500.)
    
# may 2014 dataset

elif name == 'J1323':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1347':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e18,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e18,typ=typ,maxlim=8500.)
    

elif name == 'J1446':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8000.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8000.)


elif name == 'J1605':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1606':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    

elif name == 'J1619':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4900.,bias=1e8,wid=wid,typ=typ,maxlim=8500.)


elif name == 'J2228':
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4700.,bias=1e8,wid=wid,typ=typ,maxlim=8000.)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,mask=np.array([[7580,7700],[6860,6900]]),nfit=12,bg='polynomial',lim=4700.,bias=1e8,wid=wid,typ=typ,maxlim=8000.)
    


lp,trace,dic,_=result
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/'+filename+'_chain.dat',trace[200:].reshape((trace[200:].shape[0]*trace.shape[1],trace.shape[-1])),header=' lens velocity -- lens dispersion -- source velocity -- source dispersion \n -500,500, 0,500, -500,500, 0,500')

pl.suptitle(name)
#pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/'+filename+'.pdf')
#pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/'+filename+'.png')
#pl.show()

from tools import iCornerPlotter as i
i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/'+filename+'_chain.dat,blue'])
pl.show()
#pl.close('all')

pl.figure()
pl.plot(lp)
pl.show()
#pl.close('all')
