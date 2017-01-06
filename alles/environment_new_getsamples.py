import sqlutil, numpy as np, pylab as pl, pyfits as py
from astLib import astCalc

def clip(arr,nsig=4.5):
    a = np.sort(arr.ravel())
    a = a[a.size*0.001:a.size*0.999]
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s

# now go into photo catalogue and find objects near sources.
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
radec = np.load('/data/ljo31/Lens/LensParams/radec.npy')[()]
names = sz.keys()
names.sort()
names = np.delete(names,6)

#names =[ 'J0837','J0901']
eels,nums = [], []
for name in names:
    rasrc,decsrc = radec[name]
    z = sz[name]
    scale = astCalc.da(z)*1e3*np.pi/180.
    degscale = 1.5e3 / scale
    print name, 'going into sql', degscale
    tab = sqlutil.get('select pt.type,pt.ra,pt.dec,p.z,p.zErr,p.chisq,p.nnSpecz from sdssdr9.phototag as pt, sdssdr9.photoz as p where q3c_join('+str(rasrc)+','+str(decsrc)+',ra,dec,'+str('%.2f'%degscale)+') and (z-'+str(sz[name])+')*(z-'+str(sz[name])+')<0.01*0.01 and p.objID=pt.objID and pt.type=3 limit 1000',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
    arr = np.zeros((len(tab)-1,len(tab[0])))
    for ii in range(len(tab)-1):
        arr[ii] = tab[ii+1]
    ra,dec,z,dz,chi2,specz = arr
    cls = tab[0]
    eels.append([cls,arr])
    nums.append(arr[:,cls==3].shape[1])
    print nums[-1]

# none of them have that many companions. Now go and find similar objects to the eels (similar colours and redshifts, but elsewhere) and see what the distribution is!
from spsmodels import getsdssmags as sdss

compeels = []
'''for name in names:
    if name in ['J0837','J0901','J0913','J1125','J1144','J1323','J1218','J1347','J1446','J1605']:
        compeels.append([[3]*1000,np.loadtxt('/data/ljo31b/EELs/sdss/sdss_eels_twins'+name+'.dat')])
    if name == 'J1248':
        continue
    z = sz[name]
    modelmags = sdss.getmag(name)
    G,R,I,Z=modelmags['g_SDSS'], modelmags['r_SDSS'],modelmags['i_SDSS'], modelmags['z_SDSS']
    print name, 'going into sql'
    tab = sqlutil.get('select pt.type,pt.ra,pt.dec,pt.ModelMag_r,pt.ModelMag_i,pt.ModelMag_z,pt.ModelMag_g,p.z,p.zErr,p.chisq from sdssdr9.phototag as pt, sdssdr9.photoz as p where (z-'+str(sz[name])+')*(z-'+str(sz[name])+')<0.01*0.01 and (pt.ModelMag_g-'+str(G)+')*(pt.ModelMag_g-'+str(G)+')<0.2 and (pt.ModelMag_r-'+str(R)+')*(pt.ModelMag_r-'+str(R)+')<0.2 and (pt.ModelMag_i-'+str(I)+')*(pt.ModelMag_i-'+str(I)+')<0.2 and (pt.ModelMag_z-'+str(Z)+')*(pt.ModelMag_z-'+str(Z)+')<0.2 and p.objID=pt.objID and pt.type=3 and p.chisq<4 and p.chisq>0.5 limit 1000',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
    arr = np.zeros((len(tab)-1,len(tab[0])))
    for ii in range(len(tab)-1):
        arr[ii] = tab[ii+1]
    ra,dec,R,I,Z,G,z,dz,chisq = arr
    cls = tab[0]
    compeels.append([cls,arr])
    print arr.shape
ceels = [compeels[ii][1] for ii in range(len(compeels))]

for ii in range(len(names)):
    name = names[ii]
    np.savetxt('/data/ljo31b/EELs/sdss/sdss_eels_twins'+name+'.dat',ceels[ii],header='ra,dec,R,I,Z,G,z,dz,chisq')'''
ceels = []
for ii in range(len(names)):
    name = names[ii]
    ceels.append(np.loadtxt('/data/ljo31b/EELs/sdss/sdss_eels_twins'+name+'.dat'))


# now for each eel, for each twin, go and redo the first step. How do these compare with the environmemts of the true EELs?
allnums =[]
for ii in range(len(names)):
    subnums = []
    name = names[ii]
    print name
    table = ceels[ii]
    #print 'check table shape', table.shape # check
    RA,DEC,R,I,Z,G,REDZ,DREDZ,CHISQ = table
    #print Z.size
    for jj in range(Z.size):
        RA,DEC,R,I,Z,G,REDZ,DREDZ,CHISQ = table[:,jj]
        #print RA,DEC,R,I,Z,G,REDZ,DREDZ,CHISQ
        scale = astCalc.da(REDZ)*1e3*np.pi/180.
        degscale = 1.5e3 / scale
        #print name, 'going into sql'#,degscale,ii,jj
        tab = sqlutil.get('select pt.type,pt.ra,pt.dec,p.z,p.zErr,p.chisq,p.nnSpecz from sdssdr9.phototag as pt, sdssdr9.photoz as p where q3c_join('+str(RA)+','+str(DEC)+',ra,dec,'+str('%.2f'%degscale)+') and (z-'+str(REDZ)+')*(z-'+str(REDZ)+')<0.01*0.01 and p.objID=pt.objID and pt.type=3 limit 10000',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
        arr = np.zeros((len(tab)-1,len(tab[0])))
        for kk in range(len(tab)-1):
            arr[kk] = tab[kk+1]
        cls = tab[0] 
        subnums.append(arr[:,cls==3].shape[1])
        #print subnums[-1]
    allnums.append(subnums)

from scipy.stats import ks_2samp as ks
for ii in range(len(names)):
    name = names[ii]
    pl.figure()
    pl.title(name)
    pl.hist(allnums[ii],bins=np.arange(0,30,1),histtype='stepfilled')
    pl.axvline(nums[ii],color='k')
    pl.xlabel('number of nearby galaxies')
    pl.show()
    print name, ks(allnums[ii],np.array([nums[ii]]))[1]

#print ks(np.array(np.array(allnums).ravel(),np.array(nums).ravel()))

