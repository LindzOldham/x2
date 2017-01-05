import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter_esi_indous_vdfit import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy.interpolate import splrep, splev
from scipy import sparse
import VDfit


    
sigsci = lambda wave: 20.40
t1 = VDfit.INDOUS(sigsci)
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
    srclim = 3650.*(1.+zs)+40.

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

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','jul2016_J1605','J1619','jul2016_J2228']
wids = [['J0837','0.31_source'],['J0901','0.31_source'],['J0913','0.31_source'],['J1125','0.31_source'],['J1144','0.31_source'],['J1218','0.31_source'],['J1323','0.3_lens'],['J1347','0.3_lens'],['J1446','0.31_source'],['jul2016_J1605','0.31_source'],['J1619','0.30_lens'],['jul2016_J2228','1.10_source']]
wids = dict(wids)
dir = '/data/ljo31/Lens/TeXstuff/FP_paper/'
'''
# kineamtics and plain spectra
for name in names:
    wid,typ = wids[name].split('_')
    print wid, typ
    print name, wid, typ
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    if name[:3] == 'jul':
        run(lz[name[8:]],sz[name[8:]],fit=False,read=True,File = filename,typ=typ,wid=wid)
        pl.figtext(0.065,0.88,name[8:],fontsize=30)
    else:
        run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
        pl.figtext(0.065,0.88,name,fontsize=30)
    pl.savefig(dir+name+'.pdf')
    pl.close()
    #pl.show()
'''

# just kinematics
for name in names:
    wid,typ = wids[name].split('_')
    print wid, typ
    print name, wid, typ
    filename = name+'_'+wid+'_'+typ+'_esi_indous_vdfit'
    if name[:3] == 'jul':
        run(lz[name[8:]],sz[name[8:]],fit=False,read=True,File = filename,typ=typ,wid=wid)
        pl.figtext(0.14,0.88,name[8:],fontsize=30)
        pl.savefig(dir+name[8:]+'2.pdf')
    else:
        run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
        pl.figtext(0.14,0.88,name,fontsize=30)
        pl.savefig(dir+name+'2.pdf')
    pl.close()
    #pl.show()
