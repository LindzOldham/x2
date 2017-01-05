import numpy as np, pylab as pl, pyfits as py

light = 299792.458
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]
sznew,lznew = [],[]

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1248','J1323','J1347','J1446','J1605','J1606','J1619','J2228']
keys = ['lens velocity','source velocity','lens dispersion','source dispersion']
MM =np.zeros((len(names), len(keys)))
LL,UU = MM*0.,MM*0.
n=0
for name in names:
    zl,zs = lz[name],sz[name]
    #print zl,zs
    print name, '& $',
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+name)
    lp,trace,dic,_ = result
    trace = trace[150:]
    L,M,U = [],[],[]
    for key in keys:
        dic[key] = dic[key][150:]
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
np.save('/data/ljo31b/EELs/esi/kinematics/inference/results',[LL,MM,UU])
np.save('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated',dict(sznew))
np.save('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated',dict(lznew))

