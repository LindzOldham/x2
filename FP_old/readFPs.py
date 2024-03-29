import numpy as np, pylab as pl, pyfits as py
import cPickle

xx,yy,zz,dxx,dyy,dzz = np.load('/data/ljo31b/EELs/esi/kinematics/FP_EELs.npy').T
xxc,yyc,zzc,dxxc,dyyc,dzzc = np.load('/data/ljo31b/EELs/esi/kinematics/FP_Coma.npy').T
rho = 0.99

## coma result
result = np.load('/data/ljo31b/EELs/FP/inference/FP_Coma')
lp,trace,dic,_ = result
d,l,u = [],[],[]
for key in dic.keys():
    dic[key] = dic[key][1500:]
    f = dic[key].flatten()
    up,lo,med = np.percentile(f,84)-np.percentile(f,50),np.percentile(f,50)-np.percentile(f,16),np.percentile(f,50)
    d.append([key,med])
    u.append([key,up])
    l.append([key,lo])

d,u,l=dict(d),dict(u),dict(l)
for key in d.keys():
    print key, '%.2f'%d[key], '%.2f'%l[key],'%.2f'%u[key]

np.save('/data/ljo31b/EELs/FP/inference/FP_Coma_results',np.column_stack((d,u,l)))

## EELs result
result = np.load('/data/ljo31b/EELs/FP/inference/FP')
lp,trace,dic,_ = result
d,l,u = [],[],[]
for key in dic.keys():
    dic[key] = dic[key][1000:]
    f = dic[key].flatten()
    up,lo,med = np.percentile(f,84)-np.percentile(f,50),np.percentile(f,50)-np.percentile(f,16),np.percentile(f,50)
    d.append([key,med])
    u.append([key,up])
    l.append([key,lo])

d,u,l=dict(d),dict(u),dict(l)
for key in d.keys():
    print key, '%.2f'%d[key], '%.2f'%l[key],'%.2f'%u[key]

np.save('/data/ljo31b/EELs/FP/inference/FP_results',np.column_stack((d,u,l)))

