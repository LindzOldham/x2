import numpy as np
import pylab as pl

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]

for name in sz.keys():
    # orig
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+name)
    lp,trace,dic,_=result
    n=trace.shape[0]
    l,s = dic['lens dispersion'][n-200:].ravel(),dic['source dispersion'][n-200:].ravel() 
    minl,maxl,mins,maxs = np.min(l)-50,np.max(l)+50,np.min(s)-50,np.max(s)+50
    with open( '/data/ljo31b/EELs/esi/kinematics/inference/cornerplot_chains/'+name+'.dat','wb') as f:
        f.write(b'# lens dispersion -- source dispersion \n#'+str(minl)+','+str(maxl)+','+str(mins)+','+str(maxs)+' \n')
        np.savetxt(f,np.column_stack((l,s)))
    # wide
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/wide/'+name)
    lp,trace,dic,_=result
    n=trace.shape[0]
    l,s = dic['lens dispersion'][n-200:].ravel(),dic['source dispersion'][n-200:].ravel() 
    minl,maxl,mins,maxs = np.min(l)-50,np.max(l)+50,np.min(s)-50,np.max(s)+50
    with open('/data/ljo31b/EELs/esi/kinematics/inference/cornerplot_chains/wide_'+name+'.dat','wb') as f:
        f.write(b'# lens dispersion -- source dispersion \n#'+str(minl)+','+str(maxl)+','+str(mins)+','+str(maxs)+' \n')
        np.savetxt(f,np.column_stack((l,s)))


from tools import iCornerPlotter as i
for name in sz.keys():
    i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/cornerplot_chains/'+name+'.dat,CornflowerBlue','/data/ljo31b/EELs/esi/kinematics/inference/cornerplot_chains/wide_'+name+'.dat,Crimson'])
    pl.figtext(0.65,0.75,name,fontsize=40)
    pl.savefig('/data/ljo31b/EELs/esi/kinematics/inference/cornerplot_chains/'+name+'.png')
pl.show()
