import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
from numpy import cos, sin, tan
import cPickle


table = py.open('/data/ljo31/Lens/LensParams/LensinggalaxyPhot_2src.fits')[1].data
table2 = py.open('/data/ljo31/Lens/LensParams/LensinggalaxyPhot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckGalPhot_2src.fits')[1].data
table2_k = py.open('/data/ljo31/Lens/LensParams/KeckGalPhot_1src.fits')[1].data
lumv=np.zeros(12)
v,i,k,dv,di,dk,name,Re,dRe = lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.
v[:9],i[:9],k[:9],Re[:9] = table['mag v'],table['mag i'],table_k['mag k'],table['Re v']
dv[:9],di[:9],dk[:9],dRe[:9] = table['mag v hi'],table['mag i hi'],table_k['mag k hi'],table['Re v hi']
dk[9],dk[10],dk[11] = table2_k['mag k lo'][0], table2_k['mag k lo'][1], table2_k['mag k lo'][5]
di[9],di[10],di[11] = table2['mag i lo'][0], table2['mag i lo'][1], table2['mag i lo'][5]
dv[9],dv[10],dv[11] = table2['mag v lo'][0], table2['mag v lo'][1], table2['mag v lo'][5]
Re[9],Re[10],Re[11] = table2['Re v'][0], table2['Re v'][1], table2['Re v'][5]
dRe[9],dRe[10],dRe[11] = table2['Re v hi'][0], table2['Re v hi'][1], table2['Re v hi'][5]
k[9],k[10],k[11] = table2_k['mag k'][0], table2_k['mag k'][1], table2_k['mag k'][5]
i[9],i[10],i[11] = table2['mag i'][0], table2['mag i'][1], table2['mag i'][5]
v[9],v[10],v[11] = table2['mag v'][0], table2['mag v'][1], table2['mag v'][5]
name = np.concatenate((table['name'],np.array(['J0837','J0901','J1218'])))
sort = np.argsort(name)
v,I,k,dv,di,dk,name,Re,dRe = v[sort],i[sort],k[sort],dv[sort],di[sort],dk[sort],name[sort],Re[sort],dRe[sort]
dvk = np.sqrt(dv**2.+dk**2.)
dvi = np.sqrt(dv**2.+di**2.)
vi,vk=v-I,v-k

table = py.open('/data/ljo31/Lens/LensParams/LensinggalaxyPhot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckGalPhot_1src.fits')[1].data
v1,i1,k1, dv1,di1,dk1 = table['mag v'], table['mag i'],table_k['mag k'], table['mag v hi'], table['mag i hi'],table_k['mag k hi']
dvk1 = np.sqrt(dv1**2.+dk1**2.)
dvi1 = np.sqrt(dv1**2.+di1**2.)
vi1,vk1=v1-i1,v1-k1

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{C{1.0cm}|C{1.7cm}C{1.7cm}C{1.7cm}C{1.7cm}C{1.7cm}C{1.7cm}C{1.7cm}C{1.6cm}C{1.6cm}C{1.6cm}}\hline'
print r'name & $v$ (1src) &  $v$ (2src) & $i$ (1src) & $i$ (2src) & $k$ (1src) & $k$ (2src) & $v-i$ (src) & $v-i$ (2src) & $v-k$ (1src) & $v-k$ (2src) \\\hline'
for i in range(len(name)):
    print name[i], '& $', '%.2f'%v[i], r'\pm', '%.2f'%dv[i],'$ & $','%.2f'%v1[i], r'\pm', '%.2f'%dv1[i],'$ & $', '%.2f'%I[i], r'\pm', '%.2f'%di[i],'$ & $', '%.2f'%i1[i], r'\pm', '%.2f'%di1[i],'$ & $', '%.2f'%k[i], r'\pm', '%.2f'%dk[i],'$ & $', '%.2f'%k1[i], r'\pm', '%.2f'%dk1[i],'$ & $', '%.2f'%vi[i],'$ & $', '%.2f'%vi1[i],'$ & $', '%.2f'%vk[i],'$ & $', '%.2f'%vk1[i], r'$ \\'
