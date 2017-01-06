import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import cPickle
from tools.EllipsePlot import *
from astLib import astCalc

m,l,h = np.load('/data/ljo31/Lens/LensParams/Structure_2src_huge_new.npy')
m1,l1,h1 = np.load('/data/ljo31/Lens/LensParams/Structure_1src_huge_new.npy')
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
name = m.keys()

# first, focus on 211 models
re,n,dre,dn = [],[],[],[]
q = []
names = []
for name in m.keys():
    Da = astCalc.da(sz[name])
    scale = Da*1e3*np.pi/180./3600.
    n.append(m1[name]['Source 1 n'])
    re.append(m1[name]['Source 1 re']*0.05*scale)
    dn.append(l1[name]['Source 1 n'])
    dre.append(l1[name]['Source 1 re']*0.05*scale)
    q.append(m1[name]['Source 1 q'])
    names.append(name)

n,re,dn,dre =np.array(n), np.array(re), np.array(dn), np.array(dre)
q = np.array(q)
# compare with Simard
cat = py.open('/data/ljo31b/EELs/catalogues/joinedsimtabs.fits')[1].data
ng = cat['ng3'] # pure sersic fit
pl.figure()
pl.hist(ng,50,normed=1,histtype='stepfilled',label='Simard+11')
#pl.hist(n,10,normed=1,histtype='stepfilled',label='EELs nuggets')
for ii in range(len(n)-1):
    pl.axvline(n[ii],color='Black',ls='--',lw=3)
pl.axvline(n[-1],color='Black',ls='--',label='EELs',lw=3)
pl.xlim([0.5,8])
pl.xlabel('$n$ (X models)')
pl.legend(loc='upper left')
pl.savefig('/data/ljo31/Lens/TeXstuff/paper/nhistHUGE.pdf')

# plot table
for ii in range(len(n)):
    print names[ii], '& $', '%.2f'%n[ii], '\pm', '%.2f'%dn[ii], '$'

# 2-component models
re1,re2,n1,n2,dre1,dre2,dn1,dn2 = [],[],[],[],[],[],[],[]
for name in m.keys():
    print name
    conditions = ['J0837' in name, 'J0901' in name, 'J1218' in name]
    Da = astCalc.da(sz[name])
    scale = Da*1e3*np.pi/180./3600.
    if any(conditions):
        print 'bad'
        n1.append(0)
        re1.append(0)
        dn1.append(0)
        dre1.append(0)
        n2.append(0)
        re2.append(0)
        dn2.append(0)
        dre2.append(0)
        continue
    n1.append(m[name]['Source 1 n'])
    re1.append(m[name]['Source 1 re']*0.05*scale)
    dn1.append(l[name]['Source 1 n'])
    dre1.append(l[name]['Source 1 re']*0.05*scale)
    ###
    n2.append(m[name]['Source 2 n'])
    re2.append(m[name]['Source 2 re']*0.05*scale)
    dn2.append(l[name]['Source 2 n'])
    dre2.append(l[name]['Source 2 re']*0.05*scale)
  
# order: disk is n1, bulge is n2
for ii in range(len(n1)):
    a,b = n1[ii],n2[ii]
    n1[ii], n2[ii] = np.sort((a,b))
    print '%.2f'%a,'%.2f'%b,'%.2f'%n1[ii],'%.2f'%n2[ii]

pl.figure()
pl.hist(ng,50,normed=1,histtype='stepfilled',label='Simard+11')
for ii in range(len(n1)-1):
    pl.axvline(n1[ii],color='FireBrick',ls='--',lw=3)
    pl.axvline(n2[ii],color='Black',ls='--',lw=3)
pl.axvline(n1[-1],color='FireBrick',ls='--',label='disk',lw=3)
pl.axvline(n2[-1],color='Black',ls='--',label='bulge',lw=3)
pl.xlim([0.5,8])
pl.xlabel('$n$ (Y models)')
pl.legend(loc='upper left')
#pl.savefig('/data/ljo31/Lens/TeXstuff/paper/nhistYmodelsHUGE.pdf')

    

print r'\begin{tabular}{c|ccc}\hline'
print r'lens & $n$ (X model) & $n_{disk} (Y model) & $n_{bulge}$ (Y model) \\\hline'
# total table
for ii in range(len(n)):
    print names[ii], '& $', '%.2f'%n[ii], '\pm', '%.2f'%dn[ii], '$ & $', '%.2f'%n1[ii], '\pm', '%.2f'%dn1[ii], '$ & $', '%.2f'%n2[ii], '\pm', '%.2f'%dn2[ii], r'$ \\'
    

PpS = cat['PpS1'] # prob B+D is not required
BT = cat['__B_T_g1']

# also look at 2-component models. What is the distribution of Sersic indices there? And what fraction NEED two-component sources? 
# What about the non-nugget sources? They are only marginally less compact.
BTeels = np.array([0.43,0.11,0.61,0.33,0.57,0.76,0.21,0.41,0.51])
BTeels = np.array([0.72,0.71,0.61,0.47,0.72,0.26,0.44,0.52])
pl.figure()
pl.hist(BT[PpS<0.3],25,normed=1,histtype='stepfilled',label='Simard+11 \n $p(Ps)<0.3$',alpha=0.6)
pl.hist(BT[PpS>0.3],25,normed=1,histtype='stepfilled',label='Simard+11 \n $p(Ps)>0.3$',alpha=0.6)
for ii in range(len(BTeels)-1):
    pl.axvline(BTeels[ii],ls='--',lw=3,color='k')
pl.axvline(BTeels[-1],ls='--',lw=3,color='k',label='EELs')
pl.xlim([0,1])
pl.xlabel('$B/T$ (Y models)')
pl.legend(loc='upper left')
#pl.savefig('/data/ljo31/Lens/TeXstuff/paper/nhistBTeelsHUGE.pdf')
# add our BT ratios!

# finally, axis ratios / ellipticities
e = py.open('/data/ljo31b/EELs/catalogues/e3.fits')[1].data['e']
e = e[np.isnan(e)==False]
eeels = 1-q
pl.figure()
pl.hist(1-e,25,normed=1,histtype='stepfilled',label='Simard+11')
for ii in range(len(eeels)-1):
    pl.axvline(q[ii],ls='--',lw=3,color='k')
pl.axvline(q[-1],ls='--',lw=3,color='k',label='EELs')
pl.xlim([0.2,1])
pl.xlabel('$q$ (X models)')
pl.legend(loc='upper left')
pl.savefig('/data/ljo31/Lens/TeXstuff/paper/nhistelliptHUGE.pdf')

'''
n1,n2,re1,re2 = vk*0.,vk*0.,vk*0.,vk*0.
dn1,dn2,dre1,dre2 =  vk*0.,vk*0.,vk*0.,vk*0
n,re,dn,dre = vk*0.,vk*0.,vk*0.,vk*0
for name in m.keys():
    Da = astCalc.da(z[name])
    scale = Da*1e3*np.pi/180./3600.
    if name[:6] == 'Source':
        token = 'Source'
    else:
        token = 'Galaxy'
    n1[ii],n2[ii],re1[ii],re2[ii] = m[name][token+' 1 n'], m[name][token+' 2 n'], scale*0.05*m[name][token+' 1 re'], scale*0.05*m[name][token+' 2 re'] 
        

for ii in range(len(name)):
    Da = astCalc.da(z[name[ii]])
    scale = Da*1e3*np.pi/180./3600.
    if name[ii] in m.keys():
        n1[ii],n2[ii],re1[ii],re2[ii] = m[name[ii]]['Source 1 n'], m[name[ii]]['Source 2 n'], scale*0.05*m[name[ii]]['Source 1 re'], scale*0.05*m[name[ii]]['Source 2 re'] 
        dn1[ii],dn2[ii],dre1[ii],dre2[ii] = l[name[ii]]['Source 1 n'], l[name[ii]]['Source 2 n'], scale*0.05*l[name[ii]]['Source 1 re'], scale*0.05*l[name[ii]]['Source 2 re']        
    n[ii],dn[ii],re[ii],dre[ii] = m1[name[ii]]['Source 1 n'], l1[name[ii]]['Source 1 n'], scale*0.05*m1[name[ii]]['Source 1 re'], scale*0.05*l1[name[ii]]['Source 1 re'] 

# now make a table with everything on it
for ii in range(len(name)):
    if name[ii] in m.keys():
        print name[ii], '& $', '%.2f'%logR[ii], ' \pm ', '%.2f'%dlogR[ii], '$ & $', '%.2f'%logM[ii], ' \pm ', '%.2f'%dlogM[ii], '$ & $', '%.2f'%n1[ii], ' \pm ', '%.2f'%dn1[ii], '$ & $', '%.2f'%re1[ii], ' \pm ', '%.2f'%dre1[ii], '$ & $','%.2f'%n2[ii], ' \pm ', '%.2f'%dn2[ii], '$ & $', '%.2f'%re2[ii], ' \pm ', '%.2f'%dre2[ii], '$ & $', '%.2f'%n[ii], ' \pm ', '%.2f'%dn[ii], '$ & $', '%.2f'%re[ii], ' \pm ', '%.2f'%dre[ii], r'$ \\'
    else:
        print name[ii], '& $', '%.2f'%logR[ii], ' \pm ', '%.2f'%dlogR[ii], '$ & $', '%.2f'%logM[ii], ' \pm ', '%.2f'%dlogM[ii], '$ & -- & -- & -- & -- & $', '%.2f'%n[ii], ' \pm ', '%.2f'%dn[ii], '$ & $', '%.2f'%re[ii], ' \pm ', '%.2f'%dre[ii], r'$ \\'

'''
