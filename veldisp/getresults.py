import pyfits as py, pylab as pl, numpy as np
from tools import iCornerPlotter as i

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
names = sz.keys()
names.sort()
dir = '/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'
savedir = '/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/cornerplots/'
esi_indous = '_1.00_lens_esi_indous_vdfit_chain.dat'
esi_bc03 = '_1.00_lens_esi_bc03_vdfit_chain.dat'
sdss_indous = '_sdss_indous_vdfit_chain.dat'
sdss_bc03 = '_sdss_bc03_vdfit_chain.dat'
for name in names:
    if name == 'J1248':
        continue
    velL,sigL,velS,sigS = np.loadtxt(dir+name+esi_indous).T
    np.savetxt(dir+name+'_DISP'+esi_indous,np.column_stack((sigL,sigS)),header=' lens dispersion -- source dispersion \n 0,500, 0,500')
    velL,sigL,velS,sigS = np.loadtxt(dir+name+esi_bc03).T
    np.savetxt(dir+name+'_DISP'+esi_bc03,np.column_stack((sigL,sigS)),header=' lens dispersion -- source dispersion \n 0,500, 0,500')
    velL,sigL,velS,sigS = np.loadtxt(dir+name+sdss_indous).T
    np.savetxt(dir+name+'_DISP'+sdss_indous,np.column_stack((sigL,sigS)),header=' lens dispersion -- source dispersion \n 0,500, 0,500')
    velL,sigL,velS,sigS = np.loadtxt(dir+name+sdss_bc03).T
    np.savetxt(dir+name+'_DISP'+sdss_bc03,np.column_stack((sigL,sigS)),header=' lens dispersion -- source dispersion \n 0,500, 0,500')
    i.CornerPlotter([dir+name+'_DISP'+esi_indous+',blue',dir+name+'_DISP'+esi_bc03+',red',dir+name+'_DISP'+sdss_indous+',Pink',dir+name+'_DISP'+sdss_bc03+',MediumSpringGreen'])
    #i.CornerPlotter([dir+name+esi_bc03+',red',dir+name+sdss_indous+',Pink',dir+name+sdss_bc03+',MediumSpringGreen'])
    pl.figtext(0.65,0.65,name,fontsize=30)
    pl.savefig(savedir+name+'.png')
    pl.show()
