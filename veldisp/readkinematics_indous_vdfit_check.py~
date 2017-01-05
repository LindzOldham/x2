import numpy as np, pylab as pl, pyfits as py

import sys
#ap = sys.argv[1]
#print ap

'''light = 299792.458
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
sznew,lznew = [],[]

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']

keys = ['lens velocity','source velocity','lens dispersion','source dispersion']
MM =np.zeros((len(names), len(keys)))
LL,UU = MM*0.,MM*0.
n=0
for name in names:
    zl,zs = lz[name][0],sz[name][0]
    #print zl,zs
    print name, '& $',
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+name+'_1.00_lens_esi_bc03_vdfit')
    lp,trace,dic,_ = result
    
    trace = trace[200:]
    

    pl.figure()
    pl.plot(lp[200:])
    pl.title(name)
    pl.show()

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
np.save('/data/ljo31b/EELs/esi/kinematics/inference/results_1.00_lens_bc03_vdfit',[LL,MM,UU])
np.save('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_lens_bc03_vdfit',dict(sznew))
np.save('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_1.00_lens_bc03_vdfit',dict(lznew))
'''
# now source

light = 299792.458

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
sznew,lznew = [],[]

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']

keys = ['lens velocity','source velocity','lens dispersion','source dispersion']
MM =np.zeros((len(names), len(keys)))
LL,UU = MM*0.,MM*0.
n=0
for name in names:
    zl,zs = lz[name][0],sz[name][0]
    #print zl,zs
    print name, '& $',
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+name+'_1.50_source_esi_bc03_vdfit')
    if name == 'J1605' or name == 'J2228':
        result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/jul2016_'+name+'_1.50_source_esi_bc03_vdfit')
    lp,trace,dic,_ = result
     
    trace = trace[200:]


    pl.figure()
    pl.plot(dic['source dispersion'][200:])
    pl.title(name)
    pl.show()

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
np.save('/data/ljo31b/EELs/esi/kinematics/inference/results_1.50_source_bc03_vdfit_jul2016',[LL,MM,UU])
np.save('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.50_source_bc03_vdfit_jul2016',dict(sznew))
np.save('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_1.50_source_bc03_vdfit_jul2016',dict(lznew))

'''

# now do the same for SDSS!!!

#############
############
#############

light = 299792.458
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
sznew,lznew = [],[]

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']

keys = ['lens velocity','source velocity','lens dispersion','source dispersion']
MM =np.zeros((len(names), len(keys)))
LL,UU = MM*0.,MM*0.
n=0
for name in names:
    zl,zs = lz[name][0],sz[name][0]
    #print zl,zs
    print name, '& $',
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+name+'_sdss_bc03_vdfit')
    lp,trace,dic,_ = result
    
    trace = trace[200:]
    

    pl.figure()
    pl.plot(lp[200:])
    pl.title(name)
    pl.show()

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
np.save('/data/ljo31b/EELs/esi/kinematics/inference/results_sdss_bc03_vdfit',[LL,MM,UU])
np.save('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_sdss_bc03_vdfit',dict(sznew))
np.save('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_sdss_bc03_vdfit',dict(lznew))

# now indous

light = 299792.458

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
sznew,lznew = [],[]

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']

keys = ['lens velocity','source velocity','lens dispersion','source dispersion']
MM =np.zeros((len(names), len(keys)))
LL,UU = MM*0.,MM*0.
n=0
for name in names:
    zl,zs = lz[name][0],sz[name][0]
    #print zl,zs
    print name, '& $',
    try:
        result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+name+'_sdss_indous_vdfit')
    except:
        result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/vdfit/'+name+'_sdss_indous_vdfit')
    lp,trace,dic,_ = result
     
    trace = trace[200:]

    if name == 'J0837':
        trace = trace[100:]
        lp = lp[100:]
        for key in dic.keys():
            dic[key] = dic[key][100:]

    elif name == 'J1323':
        lp = lp[:,lp[-1]>-10200]
        trace = trace[:,lp[-1]>-10200]
        for key in dic.keys():
            dic[key] = dic[key][:,lp[-1]>-10200]

    pl.figure()
    pl.plot(lp[200:])
    pl.title(name)
    pl.show()

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
np.save('/data/ljo31b/EELs/esi/kinematics/inference/results_sdss_indous_vdfit',[LL,MM,UU])
np.save('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_sdss_indous_vdfit',dict(sznew))
np.save('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_1.50_sdss_indous_vdfit',dict(lznew))
'''
