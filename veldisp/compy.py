import numpy as np, pylab as pl
from tools import iCornerPlotter as i

for name in ['J0837','J0913','J1125','J1144','J1218']:
    i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_'+name+'_8500_sky,blue','/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_'+name+'_9000_sky,red'])
    pl.figtext(0.65,0.65,name,fontsize=30)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/9000_8500_'+name+'corner.png')

i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_J0837_8500_sky,blue','/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_J0837_9000_sky,red','/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_J0837_9500_sky,LightGray','/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_J0837_8000_sky,Purple'])
    pl.figtext(0.65,0.65,name,fontsize=30)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/9000_8500_'+name+'corner.png')
