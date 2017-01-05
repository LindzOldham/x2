import numpy as np, pylab as pl, pyfits as py
from tools import iCornerPlotter as i

import sys
#ap = sys.argv[1]
#print ap

light = 299792.458
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]
sznew,lznew = [],[]

names = ['J0837','J0901','J0913','J1125','J1144','J1218']#,'J1323','J1347','J1446','J1605','J1606','J1619','J2228']
mwa_l = [0,158,202,0,252,0]
mwa_s = [0,310,205,0,293,0]

keys = ['lens velocity','source velocity','lens dispersion','source dispersion']
MM =np.zeros((len(names), len(keys)))
LL,UU = MM*0.,MM*0.
n=0
for name in names:
    zl,zs = lz[name],sz[name]
    #print zl,zs
    print name, '& $',
    #result = np.load('/data/ljo31b/EELs/esi/kinematics/trials/sdss_'+name)
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/notvar_stitch2_sdss_'+name)
    lp,trace,dic,_ = result
    oresult = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/final/'+name+'_1.00_lens')
    olp,otrace,odic,_ = oresult

    if name == 'J0913':
        otrace = otrace[:,olp[-1]>-18000]
        for key in odic.keys():
            odic[key] = odic[key][:,olp[-1]>-18000]
        olp = olp[:,olp[-1]>-18000]

    #np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/notvar_stitch2_sdss_chain_'+name,trace[200:].reshape((trace[200:].shape[0]*trace.shape[1],trace.shape[-1])))
    #np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_'+name,otrace[200:].reshape((otrace[200:].shape[0]*otrace.shape[1],otrace.shape[-1])))
    
        i.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/esi_chain_'+name+',blue','/data/ljo31b/EELs/esi/kinematics/inference/notvar_stitch2_sdss_chain_'+name+',red'])
    pl.figtext(0.65,0.65,name,fontsize=30)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'corner.png')

   

    '''pl.figure()
    pl.subplot(211)
    pl.hist(dic['lens dispersion'][200:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.hist(odic['lens dispersion'][300:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.axvline(mwa_l[n],color='k')
    pl.xlim([5,500])
    pl.xlabel('lens dispersion')
    pl.subplot(212)
    pl.hist(dic['source dispersion'][200:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.hist(odic['source dispersion'][300:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.axvline(mwa_s[n],color='k')
    pl.xlabel('source dispersion')
    pl.xlim([5,500])
    pl.suptitle(name)
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+name+'HIST.png')'''
    trace = trace[200:]
    
    ### velocities should at least match!
    '''pl.figure()
    pl.subplot(211)
    pl.hist(dic['lens velocity'][200:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.hist(odic['lens velocity'][300:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    #pl.xlim([5,500])
    pl.xlabel('lens velocity')
    pl.subplot(212)
    pl.hist(dic['source velocity'][200:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.hist(odic['source velocity'][300:].ravel(),30,alpha=0.5,histtype='stepfilled',edgecolor='none',normed=True)
    pl.xlabel('source velocity')
    #pl.xlim([5,500])
    pl.suptitle(name)'''


    #pl.figure()
    #pl.plot(lp[200:])
    #pl.title(name)
    #pl.show()

    L,M,U = [],[],[]
    for key in keys:
        dic[key] = dic[key][200:]
        f = dic[key].reshape((trace.shape[0]*trace.shape[1]))
        lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
        L.append(lo)
        M.append(med)
        U.append(up)
        if key == 'lens dispersion':
            print '%.2f'%med, '\pm', '%.2f'%lo, '$ & $',
        elif key == 'source dispersion':
            print '%.2f'%med, '\pm', '%.2f'%lo, r'$ \\'
        elif key == 'lens velocity':
            zl += med/light
            dzl1,dzl2 = lo/light, up/light
            print '%.3f'%zl, '$ & $',
        elif key == 'source velocity':
            zs += med/light
            dzs1,dzs2 = lo/light, up/light
            print '%.3f'%zs, '$ & $',
    #print zl,zs
    sznew.append([name,[zs,dzs1,dzs2]])
    lznew.append([name,[zl,dzl1,dzl2]])
    LL[n] = L
    MM[n] = M
    UU[n] = U
    n+=1


vl,sl,vs,ss = MM.T
np.save('/data/ljo31b/EELs/esi/kinematics/inference/results_sdss_stitch2',[LL,MM,UU])
np.save('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_sdss_stitch2',dict(sznew))
np.save('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_sdss_stitch2',dict(lznew))


