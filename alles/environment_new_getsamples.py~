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

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
dist = []
for name in sz.keys():
    #print name
    tab = sqlutil.get('select class,ra,dec,z,zErr,dered_g,dered_i,dered_r,dered_z from sdssdr12.specphotoall where (z-'+str(sz[name])+')*(z-'+str(sz[name])+')<0.004*0.004 limit 500',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
    arr = np.zeros((len(tab)-1,500))
    for i in range(len(tab)-1):
        arr[i] = tab[i+1]
    ra,dec,redshift,zErr,g,i,r,z = arr
    cls = tab[0]
    idx = np.where((np.isnan(g)==False) & (np.isnan(i)==False) & (np.isnan(r)==False) & (np.isnan(z)==False) & (zErr<2e-4) & (cls=='GALAXY'))
    gr,ri,iz = (g-r)[idx],(r-i)[idx],(i-z)[idx]
    #pl.figure()
    #pl.hist(gr,np.arange(-0.5,3.5,0.1),label='g-r',alpha=0.5,histtype='stepfilled')
    #pl.hist(ri,np.arange(-0.5,3.5,0.1),label='r-i',alpha=0.5,histtype='stepfilled')
    #pl.hist(iz,np.arange(-0.5,3.5,0.1),label='i-z',color='DarkOrange',alpha=0.5,histtype='stepfilled')
    #pl.title(name)
    #pl.xlabel('colour')
    #pl.legend(loc='upper right')
    m1,m2,m3 = np.asarray(clip(gr)), np.asarray(clip(ri)), np.asarray(clip(iz))
    dist.append((name,np.concatenate((m1,m2,m3))))
    
dist = dict(dist)
pile = np.zeros((13,7))
for i in range(13):
    name = sz.keys()[i]
    redz = sz[name]
    gr,dgr,ri,dri,iz,diz = dist[name]
    pile[i] = redz,gr,dgr,ri,dri,iz,diz

#pl.figure()
#redz,gr,dgr,ri,dri,iz,diz = pile.T
#pl.plot(redz,dgr,'o')
#pl.plot(redz,dri,'o')
#pl.plot(redz,diz,'o')
# this shows there exist weak correlations with redshift - gr decreases with increasing redshift, ri increases with redshft, iz increases extremely slowly with redshift! ri is the best indicator
# sigmas all increase with redshift - as things are younger and so at a wider range of colours.

# now go into photo catalogue and find objects near sources.
radec = np.load('/data/ljo31/Lens/LensParams/radec.npy')[()]
Nws = []
Nw2 = []
for name in sz.keys():
    rasrc,decsrc = radec[name]
    z = sz[name]
    scale = astCalc.da(z)*1e3*np.pi/180.
    degscale = 1.5e3 / scale
    tab = sqlutil.get('select type,ra,dec,dered_g,dered_i,dered_r,dered_z from sdssdr9.photoobjall where q3c_join('+str(rasrc)+','+str(decsrc)+',ra,dec,'+str('%.2f'%degscale)+') limit 3000',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
    arr = np.zeros((len(tab)-1,len(tab[0])))
    for ii in range(len(tab)-1):
        arr[ii] = tab[ii+1]
    ra,dec,g,i,r,z = arr
    cls = tab[0]
    idx = np.where((np.isnan(g)==False) & (np.isnan(i)==False) & (np.isnan(r)==False) & (np.isnan(z)==False) & (cls==3) & (g>-50) & (r>-50) & (i>-50) & (z>-50) & (g>16) & (g<26) & (r>16) & (r<26) & (i>16) & (i<26) & (z>16) & (z<26))
    gr,ri,iz = (g-r)[idx],(r-i)[idx],(i-z)[idx]
    dra, ddec = (ra-rasrc)*np.cos(0.5*(dec+decsrc)), dec-decsrc
    D = np.sqrt(dra**2. + ddec**2.)[idx]*scale
    # for now, don't select magnitudes too carefully, but do this later (if everything works!)
    GR,DGR,RI,DRI,IZ,DIZ = dist[name]
    f = np.ones(len(D))
    f[D>100] = np.exp(-(D[D>100]-100.)/30.)
    Dgr, Dri, Diz = (gr-GR)**2.,(ri-RI)**2.,(iz-IZ)**2.
    w = f*np.exp(-0.5*Dgr/DGR**2.) * np.exp(-0.5*Dri/DRI**2) * np.exp(-0.5*Diz/DIZ**2.)
    Nw = w.sum()
    print name,Nw
    Nws.append((name,Nw))
    Nw2.append(Nw)

Nwdict = dict(Nws)
np.save('/data/ljo31b/EELs/environments/Nwdict',Nwdict)
# now, for each object, draw 200 samples from sdss and do the same thing...
from spsmodels import getsdssmags as sdss
Nwsamples = []
for name in sz.keys():
    z = sz[name]
    if name == 'J1248':
        continue
    scale = astCalc.da(z)*1e3*np.pi/180.
    degscale = 1.5e3 / scale
    modelmags = sdss.getmag(name)
    G,R,I,Z=modelmags['g_SDSS'], modelmags['r_SDSS'],modelmags['i_SDSS'], modelmags['z_SDSS']
    tab = sqlutil.get('select class,ra,dec,z,zErr,dered_g,dered_i,dered_r,dered_z from sdssdr12.specphotoall where (z-'+str(sz[name])+')*(z-'+str(sz[name])+')<0.004*0.004 and (dered_g-'+str(G)+')*(dered_g-'+str(G)+')<0.1 and (dered_r-'+str(R)+')*(dered_r-'+str(R)+')<0.1 and (dered_i-'+str(I)+')*(dered_i-'+str(I)+')<0.1 and (dered_z-'+str(Z)+')*(dered_z-'+str(Z)+')<0.1 limit 200',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
    cls = tab[0]
    idx = np.where(cls=='GALAXY')
    arr = np.zeros((len(tab)-1,len(tab[0])))
    print arr.shape
    for ii in range(len(tab)-1):
        arr[ii] = tab[ii+1]
    arr = arr[:,idx]
    print arr.shape
    GR,DGR,RI,DRI,IZ,DIZ = dist[name]
    Nwsample = []
    for ii in range(len(cls[idx])):
        rasrc,decsrc,_,_,_,_,_,_ = arr[:,0,ii]
        tab = sqlutil.get('select type,ra,dec,dered_g,dered_i,dered_r,dered_z from sdssdr9.photoobjall where q3c_join('+str(rasrc)+','+str(decsrc)+',ra,dec,'+str('%.2f'%degscale)+') limit 3000',db='wsdb',host='cappc127',user='lindsay_oldham',password='Crontoil1!')
        arr1 = np.zeros((len(tab)-1,len(tab[0])))
        for ii in range(len(tab)-1):
            arr1[ii] = tab[ii+1]
        ra,dec,g,i,r,z = arr1
        cls = tab[0]
        idx = np.where((np.isnan(g)==False) & (np.isnan(i)==False) & (np.isnan(r)==False) & (np.isnan(z)==False) & (cls==3) & (g>-50) & (r>-50) & (i>-50) & (z>-50) & (g>16) & (g<26) & (r>16) & (r<26) & (i>16) & (i<26) & (z>16) & (z<26))
        gr,ri,iz = (g-r)[idx],(r-i)[idx],(i-z)[idx]
        dra, ddec = (ra-rasrc)*np.cos(0.5*(dec+decsrc)), dec-decsrc
        D = np.sqrt(dra**2. + ddec**2.)[idx]*scale
        ####
        f = np.ones(len(D))
        f[D>100] = np.exp(-(D[D>100]-100.)/30.)
        Dgr, Dri, Diz = (gr-GR)**2.,(ri-RI)**2.,(iz-IZ)**2.
        w = f*np.exp(-0.5*Dgr/DGR**2.) * np.exp(-0.5*Dri/DRI**2) * np.exp(-0.5*Diz/DIZ**2.)
        Nw = w.sum()
        print name,Nw
        Nwsample.append(Nw)
    Nwsamples.append((name,Nwsample))

Nwsamples = dict(Nwsamples)
np.save('/data/ljo31b/EELs/environments/Nwsamplesdict',Nwsamples)
