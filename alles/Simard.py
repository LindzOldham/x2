import numpy as np, pylab as pl, pyfits as py

'''
cat = py.open('/data/ljo31b/EELs/catalogues/joinedsimtabs.fits')[1].data
ng = cat['ng3'] # pure sersic fit
nb = cat['nb2'] # free bulge index
PpS = cat['PpS1'] # prob B+D is not required 
# if it is required, what happens to the sizes in these systems?
Re_g, Re_bd = cat['Rhlg3'], cat['Rhlg1']

pl.figure()
pl.scatter(Re_g, Re_bd,c=PpS,s=20,edgecolor='none')
pl.xlim([0,30])
pl.ylim([0,30])
pl.colorbar()
pl.xlabel('$R_e$ / kpc (pure Sersic fit)')
pl.ylabel('$R_e$ / kpc (B+D model)')
xline=np.linspace(0,100,100)
pl.plot(xline,xline,'k')

## now repeat, only selecting objects with PpS<0.5
pl.figure()
pl.scatter(Re_g[PpS<0.3], Re_bd[PpS<0.3],c=PpS[PpS<0.3],s=40)#,edgecolor='none')
pl.xlim([0,30])
pl.ylim([0,30])
pl.colorbar()
pl.xlabel('$R_e$ / kpc (pure Sersic fit)')
pl.ylabel('$R_e$ / kpc (B+D model)')
xline=np.linspace(0,100,100)
pl.plot(xline,xline,'k')

## now repeat, only selecting objects with PpS>0.7
pl.figure()
pl.scatter(Re_g[PpS>0.7], Re_bd[PpS>0.7],c=PpS[PpS>0.7],s=40)#,edgecolor='none')
pl.xlim([0,30])
pl.ylim([0,30])
pl.colorbar()
pl.xlabel('$R_e$ / kpc (pure Sersic fit)')
pl.ylabel('$R_e$ / kpc (B+D model)')
xline=np.linspace(0,100,100)
pl.plot(xline,xline,'k')
'''

## let's do the same thing in the r band.
cat = py.open('/data/ljo31b/EELs/catalogues/joinedsimtabs.fits')[1].data
ng = cat['ng3'] # pure sersic fit
nb = cat['nb2'] # free bulge index
PpS = cat['PpS1'] # prob B+D is not required 
# if it is required, what happens to the sizes in these systems?
Re_g, Re_bd = cat['Rhlr3'], cat['Rhlr1']

pl.figure()
pl.scatter(Re_g, Re_bd,c=PpS,s=20,edgecolor='none')
pl.xlim([0,30])
pl.ylim([0,30])
pl.colorbar()
pl.xlabel('$R_e$ / kpc (pure Sersic fit)')
pl.ylabel('$R_e$ / kpc (B+D model)')
xline=np.linspace(0,100,100)
pl.plot(xline,xline,'k')

## now repeat, only selecting objects with PpS<0.5
pl.figure()
pl.scatter(Re_g[PpS<0.3], Re_bd[PpS<0.3],c=PpS[PpS<0.3],s=40)#,edgecolor='none')
pl.xlim([0,30])
pl.ylim([0,30])
pl.colorbar()
pl.xlabel('$R_e$ / kpc (pure Sersic fit)')
pl.ylabel('$R_e$ / kpc (B+D model)')
xline=np.linspace(0,100,100)
pl.plot(xline,xline,'k')

## now repeat, only selecting objects with PpS>0.7
pl.figure()
pl.scatter(Re_g[PpS>0.7], Re_bd[PpS>0.7],c=PpS[PpS>0.7],s=40)#,edgecolor='none')
pl.xlim([0,30])
pl.ylim([0,30])
pl.colorbar()
pl.xlabel('$R_e$ / kpc (pure Sersic fit)')
pl.ylabel('$R_e$ / kpc (B+D model)')
xline=np.linspace(0,100,100)
pl.plot(xline,xline,'k')

### now let's look at the Sersic indices of things that need B+D decompositions
cat = py.open('/data/ljo31b/EELs/catalogues/joinedsimtabs.fits')[1].data
ng = cat['ng3'] # pure sersic fit
nb = cat['nb2'] # free bulge index
PpS = cat['PpS1'] # prob B+D is not required 

pl.figure()
pl.hist(ng,bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='$p(pS)= [0,1]$')
pl.hist(ng[PpS<0.3],bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='$p(pS)<0.3$')
pl.xlabel('$n_g$ (pure Sersic fit)')
pl.legend(loc='upper left')

pl.figure()
pl.hist(nb[PpS<0.3],bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5)
pl.hist(nb[PpS<0.3],bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='$p(pS)<0.3$')
pl.xlabel('$n_g$ (B+D model)')
pl.legend(loc='upper left')

pl.figure()
pl.hist(ng,bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='$p(pS)= [0,1]$, pS')
pl.hist(nb[PpS<0.3],bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='$p(pS)<0.3$, nb')
pl.xlabel('$n_g$')
pl.legend(loc='upper left')


# and are BD things more disky or more bulgy?
BT = cat['__B_T_g1']
ii = np.where(PpS<0.3)
# subsample:
ng2,BT2 = ng[ii],BT[ii]
pl.figure()
pl.hist(ng2[BT2<0.5],bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='disk-dominated')
pl.hist(ng2[BT2>0.5],bins=np.linspace(0,7.8,50),normed=1,histtype='stepfilled',alpha=0.5,label='bulge-dominated')
pl.xlabel('$n_g$ (pure Sersic)')
pl.legend(loc='upper left')

