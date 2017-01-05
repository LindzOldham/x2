import numpy as np, pylab as pl

name = 'J0913'
dir = '/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'

pl.figure(figsize=(10,5))
ax1 = pl.subplot(121)
ax2 = pl.subplot(122)
for nfit in ['6','9','12']:
    try:
        result = np.load(dir+name+'_0.31_source_esi_indous_vdfit_BC_'+nfit)
    except:
        result = np.load(dir+name+'_0.31_source_esi_indous_vdfit_BC')
    lp,trace,dic,_ = result
    ax1.hist(dic['source velocity'].ravel(),30,alpha=0.5,label=nfit,normed=True,histtype='stepfilled')
    ax2.hist(dic['source dispersion'].ravel(),30,alpha=0.5,label=nfit,normed=True,histtype='stepfilled')
    ax1.legend(loc='upper left',fontsize=15,ncol=3)
    ax1.set_xlabel('source velocity')
    ax2.set_xlabel('source dispersion')
pl.show()

pl.figure(figsize=(10,5))
ax1 = pl.subplot(121)
ax2 = pl.subplot(122)
for nfit in ['6','9','12']:
    try:
        result = np.load(dir+name+'_0.31_source_esi_indous_vdfit_BC_'+nfit)
    except:
        result = np.load(dir+name+'_0.31_source_esi_indous_vdfit_BC')
    lp,trace,dic,_ = result
    ax1.hist(dic['lens velocity'].ravel(),30,alpha=0.5,label=nfit,normed=True,histtype='stepfilled')
    ax2.hist(dic['lens dispersion'].ravel(),30,alpha=0.5,label=nfit,normed=True,histtype='stepfilled')
    ax1.legend(loc='upper left',fontsize=15,ncol=3)
    ax1.set_xlabel('lens velocity')
    ax2.set_xlabel('lens dispersion')
pl.show()
