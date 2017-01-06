import numpy as np, pylab as pl, pyfits as py
from astLib import astCalc


def redtwo(name):
    if name == 'J1248' or name == 'J1619':
        return
    cat = np.load('/data/ljo31b/EELs/sex/'+name+'_cat.npy')
    v1,ev1,v2,ev2 = cat[:,5:9].T
    i1,ei1,i2,ei2 = cat[:,27:31].T
    ra,dec = cat[:,3:5].T
    stell = cat[:,-1]
    vi = v2-i2
    dvi = np.sqrt(ev2**2. + ei2**2.)
    v_src,i_src,dv_src,di_src,vi_src,dvi_src, v_lens,i_lens,dv_lens,di_lens,vi_lens,dvi_lens, k_src, dk_src, k_lens, dk_lens = vkidata[name]
    vi_src = v_src-i_src
    ra_src,dec_src = radec[name]
    z = sz[name]
    R = np.sqrt((ra-ra_src)**2. + (dec-dec_src)**2.)*3600.
    # select objects with stellarity<0.7 and R<500 kpc ~ 80 arcsec
    idx = np.where((R<160)&(vi<3)&(vi>-1.5)&(v2<25)&(stell<0.9))
    v,vi = v2[idx],vi[idx]
    # plot the red sequence?
    pl.figure()
    pl.scatter(v,vi,color='Crimson')
    pl.scatter(i_src,vi_src,color='k')
    pl.xlabel('$v$')
    pl.ylabel('$v-i$')
    pl.title(name)
    #pl.axis([15,28,-4,4])
    pl.axis([15,28,-4,4])

vkidata = np.load('/data/ljo31/Lens/LensParams/VIK_phot_211_dict.npy')[()]
names = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()].keys()
radec = np.load('/data/ljo31/Lens/LensParams/radec.npy')[()]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
for name in names:
    redtwo(name)


def redsequence():
    vkidata = np.load('/data/ljo31/Lens/LensParams/VIK_phot_211_dict.npy')[()]
    names = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()].keys()
    radec = np.load('/data/ljo31/Lens/LensParams/radec.npy')[()]
    sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
    for name in names:
        if name == 'J1248' or name == 'J1619':
            continue
        cat = np.load('/data/ljo31b/EELs/sex/'+name+'_cat.npy')
        v1,ev1,v2,ev2 = cat[:,5:9].T
        i1,ei1,i2,ei2 = cat[:,27:31].T
        ra,dec = cat[:,3:5].T
        vi = v2-i2
        dvi = np.sqrt(ev2**2. + ei2**2.)
        #ii = np.where(abs(vi)<5)
        # now get ra,dec,v-i of the source and choose similar things!
        v_src,i_src,dv_src,di_src,vi_src,dvi_src, v_lens,i_lens,dv_lens,di_lens,vi_lens,dvi_lens, k_src, dk_src, k_lens, dk_lens = vkidata[name]
        vi_src = v_src-i_src
        print vi_src
        ra_src,dec_src = radec[name]
        z = sz[name]
        R = np.sqrt((ra-ra_src)**2. + (dec-dec_src)**2.)*3600. # arcseconds
        ii = np.where(R<150) # within 150 arcseconds
        ra,dec,vi=ra[ii],dec[ii],vi[ii]
        diff = vi-vi_src
        pl.figure()
        pl.scatter(ra[abs(diff)<0.5],dec[abs(diff)<0.5],c=diff[abs(diff)<0.5],edgecolor='none',s=200)
        pl.plot(ra_src,dec_src,marker='*')
        pl.colorbar()
        pl.title(name)
        pl.xlabel('ra (deg)')
        pl.ylabel('dec (deg)')
        print len(diff[abs(diff)<0.5])

def makecats():
    names = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()].keys()
    for name in names:
        if name == 'J1248':
            continue
        catv = np.loadtxt('/data/ljo31b/EELs/sex/'+name+'_V.cat')
        cati = np.loadtxt('/data/ljo31b/EELs/sex/'+name+'_I.cat')
        print catv.shape
        rav,decv = catv[:,3],catv[:,4]
        rai,deci = cati[:,3],cati[:,4]
        dist,ind = match_lists(rav,decv,rai,deci,0.001)
        goodmatch_ind = np.isfinite(dist)
        gind = ind[goodmatch_ind]
        catv,cati = catv[goodmatch_ind],cati[ind[goodmatch_ind]]
        rav,decv,rai,deci = catv[:,3],catv[:,4],cati[:,3],cati[:,4]
        cat = np.column_stack((catv,cati))
        np.save('/data/ljo31b/EELs/sex/'+name+'_cat',cat)


def match_lists(ra1, dec1, ra2, dec2, dist):
    import scipy.spatial.kdtree, numpy
    from numpy import sin,cos,deg2rad,rad2deg,arcsin

    # crossmatches the list of objects (ra1,dec1) with
    # another list of objects (ra2,dec2) with the dist matching radius
    # the routine returns the distance to the neighbor and the list
    # of indices of the neighbor. Everything is in degrees.
    # if no match is found the distance is NaN.
    # Example
    # dist,ind=match(ra1,dec1,ra2,dec2)
    # goodmatch_ind = numpy.isfinite(dist)
    # plot(ra1[goodmatch_ind],ra2[ind][goodmatch_ind])

    # convert ra, dec to x, y, z
    cosd = lambda x : cos(deg2rad(x))
    sind = lambda x : sin(deg2rad(x))
    mindist = 2 * sind(dist/2)
    getxyz = lambda r, d: [cosd(r)*cosd(d), sind(r)*cosd(d), sind(d)]
    xyz1 = numpy.array(getxyz(ra1, dec1))
    xyz2 = numpy.array(getxyz(ra2, dec2))
    
# At the moment I'm using Python version of the KDTree instead of
# cKDTtree because there is a bug in the cKDTree
# http://projects.scipy.org/scipy/ticket/1178
    tree2 = scipy.spatial.cKDTree(xyz2.transpose())
    ret = tree2.query(xyz1.transpose(), 1, 0, 2, mindist)
    dist, ind = ret
    dist = rad2deg(2*arcsin(dist/2))
    return dist, ind
