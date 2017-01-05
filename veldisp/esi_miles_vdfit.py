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
filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'

may2014 = ['J1323','J1347','J1446','J1605','J1606','J1619','J2228']
if name in may2014:
    sigsci = lambda wave: 20.26
else:
    sigsci = lambda wave: 20.40

t1 = VDfit.MILES(sigsci)
t2 = VDfit.PICKLES(sigsci)

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]




def run(zl,zs,fit=True,read=False,File=None,mask=None,nfit=12,bg='polynomial',bias=100,restmask=None,wid=1.,typ=None):
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

    # define region of interest
    if type(zl)!=float:#>1.:
        zl,zs = zl[0],zs[0]
    print zl, zs
    lim = 3800.*(1.+zl)
    maxlim = 4800.*(1.+zs)
    srclim = 3450.*(1.+zs)+40.

    if srclim<=lim:
        srclim = lim+10.
    print 'lims: ', lim,srclim, maxlim

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(maxlim))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] 
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

    if fit:
        print 'running on ', nthreads, ' threads'
        result = finddispersion(scispec,varspec,t1,t2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),lim=lim,bg=bg,restmask=restmask,srclim=srclim,maxlim=maxlim,nthr=nthreads,bias=bias)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),lim=lim,bg=bg,restmask=restmask,maxlim=maxlim,srclim=srclim,bias=bias)
        return result
    else:
        return


if name=='J0837':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,typ=typ,wid=wid)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
    

elif name == 'J0901':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,typ=typ,wid=wid)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
    

elif name == 'J0913':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,typ=typ,wid=wid)#,nfit=12,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)#,nfit=12,bias=1e6)
    

elif name == 'J1125':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,typ=typ,wid=wid)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
    

elif name == 'J1144':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,typ=typ,wid=wid)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
    

elif name == 'J1218':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,typ=typ,wid=wid)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
    
# may 2014 dataset

elif name == 'J1323':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)#,nfit=12,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)#,nfit=12,bias=1e6)
    

elif name == 'J1347':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)#,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)#,bias=1e6)
    

elif name == 'J1446':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)#,nfit=12,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)#,nfit=12,bias=1e6)


elif name == 'J1605':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)#,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)#,bias=1e6)
    

elif name == 'J1606':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)#,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)#,bias=1e6)
    

elif name == 'J1619':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)


elif name == 'J2228':
    result = run(lz[name],sz[name],fit=True,read=False,File = filename,wid=wid,typ=typ)#,nfit=10,bias=1e6)
    result = run(lz[name],sz[name],fit=False,read=True,File = filename,wid=wid,typ=typ)#,nfit=10,bias=1e6)
    


lp,trace,dic,_=result
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+filename+'_chain.dat',trace[200:].reshape((trace[200:].shape[0]*trace.shape[1],trace.shape[-1])),header=' lens velocity -- lens dispersion -- source velocity -- source dispersion \n -500,500, 0,500, -500,500, 0,500')

pl.suptitle(name)
#pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/'+filename+'.pdf')
pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/dump/fits_esi_miles/'+filename+'.png')
#pl.show()

from tools import iCornerPlotter as i
i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+filename+'_chain.dat,blue'])
#pl.show()
#pl.close('all')

#pl.figure()
#pl.plot(lp)
pl.show()
pl.close('all')
