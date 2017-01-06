import pylab as pl, numpy as np, pyfits as py

'''tab1 = py.open('/data/ljo31/Lens/LensParams/Phot_1src_lensgals_new.fits')[1].data
names1,rev1,rei1,magv1,magi1 = tab1['name'],tab1['re v'], tab1['re i'],tab1['mag v'], tab1['mag i']
tab2 = py.open('/data/ljo31/Lens/LensParams/Phot_1src_lensgals_huge_quick.fits')[1].data
names2,rev2,rei2,magv2,magi2 = tab2['name'],tab2['re v'], tab2['re i'],tab2['mag v'],tab2['mag i']

rev3,rei3 = rev2*0.,rei2*0.
magv3,magi3 = rev2*0.,rev2*0.
for name in names2:
    ii = np.where(names1==name)
    rev3[np.where(names2==name)] = rev1[ii]
    rei3[np.where(names2==name)] = rei1[ii]
    magv3[np.where(names2==name)] = magv1[ii]
    magi3[np.where(names2==name)] = magi1[ii]


def scat(x,y):
    pl.scatter(x,y,s=30,color='SteelBlue')
    #pl.axhline(0,color='SteelBlue')


pl.figure()
scat(rev2,rev2-rev3)
pl.xlabel('$r_{e,v}$ (new)')
#pl.ylabel('$r_{e,v}$ (old)')
pl.ylabel('new - old')

pl.figure()
scat(rei2,rei2-rei3)
pl.xlabel('$r_{e,i}$ (new)')
#pl.ylabel('$r_{e,i}$ (old)')
pl.ylabel('new - old')

pl.figure()
scat(magv2,magv2-magv3)
pl.xlabel('$V$ (new)')
pl.ylabel('new - old')
pl.ylabel('new - old')


pl.figure()
scat(magi2,magi2-magi3)
pl.xlabel('$I$ (new)')
#pl.ylabel('$I$ (old)')
pl.ylabel('new - old')
'''

pdf = np.load('/data/ljo31/Lens/PDFs/212_lensgal_J1144.npy')
rev,rei = pdf[-1].T
magv,magi = pdf[1].T
muv,mui = pdf[0].T
pdf2 = np.load('/data/ljo31/Lens/PDFs/212_lensgal_huger_J1144.npy')
rev2,rei2 = pdf2[-1].T
magv2,magi2 = pdf2[1].T
muv2,mui2 = pdf2[0].T

# sizes
pl.figure()
pl.hist(rev,alpha=0.5,label='big box',histtype='stepfilled')
pl.hist(rev2,alpha=0.5,label='bigger box',histtype='stepfilled')
pl.xlabel('galaxy total $r_{e,v}$ (kpc)')
pl.legend(loc='lower right')

pl.figure()
pl.hist(rei,alpha=0.5,label='big box',histtype='stepfilled')
pl.hist(rei2,alpha=0.5,label='bigger box',histtype='stepfilled')
pl.xlabel('galaxy total $r_{e,i}$ (kpc)')
pl.legend(loc='lower left')

# magnitudes
pl.figure()
pl.hist(magv,alpha=0.5,label='big box',histtype='stepfilled')
pl.hist(magv2,alpha=0.5,label='bigger box',histtype='stepfilled')
pl.xlabel('galaxy total $V$ (mag)')
pl.legend(loc='lower right')

pl.figure()
pl.hist(magi,alpha=0.5,label='big box',histtype='stepfilled')
pl.hist(magi2,alpha=0.5,label='bigger box',histtype='stepfilled')
pl.xlabel('galaxy total $I$ (mag)')
pl.legend(loc='lower left')

#SBs
pl.figure()
pl.hist(muv,alpha=0.5,label='big box',histtype='stepfilled')
pl.hist(muv2,alpha=0.5,label='bigger box',histtype='stepfilled')
pl.xlabel('galaxy total $\mu_V$ (mag)')
pl.legend(loc='lower right')

pl.figure()
pl.hist(mui,alpha=0.5,label='big box',histtype='stepfilled')
pl.hist(mui2,alpha=0.5,label='bigger box',histtype='stepfilled')
pl.xlabel('galaxy total $\mu_I$ (mag)')
pl.legend(loc='lower left')
