import numpy as np, pylab as pl, pyfits as py

magv,magi,rmagv,rmagi,rmagb,muv,mui,rmuv,rmui,rv,ri,lumv,lumi,lumb = np.loadtxt('/data/ljo31/Lens/LensParams/PhotCat_1src.txt').T
ii=np.isfinite(lumi)
magv,magi,rmagv,rmagi,rmagb,muv,mui,rmuv,rmui,rv,ri,lumv,lumi,lumb = np.loadtxt('/data/ljo31/Lens/LensParams/PhotCat_1src.txt')[ii].T
names = ['J0837','J0901','J0913','J1125','J1144','J1323','J1347','J1446','J1605','J1606']
cols=['SteelBlue','Teal','CornflowerBlue','Crimson','SandyBrown','PaleVioletRed','LightSeaGreen','LightSlateGray','Gold','DarkTurquoise']
#names=names[ii]
#cols=cols[ii]

Lv,Lb = 10**lumv, 10**lumb
ML_chabv,ML_chabb = 1.54, 2.15
ML_salpv, ML_salpb = 2.79, 3.90
Mchabb, Mchabv, Msalpb, Msalpv = Lb*ML_chabb, Lv*ML_chabv, Lb*ML_salpb, Lv*ML_salpv
pl.figure()
for i in range(len(Lv)):
    #print names[i], cols[i]
    #pl.scatter(Mchabv[i],rv[i],marker='o',color=str(cols[i]),s=100)#,label=str(names[i]),s=100)
    pl.scatter(Msalpv[i],rv[i],marker='s',color=str(cols[i]),s=100)
pl.xscale('log')
pl.yscale('log')
#pl.loglog(Mchabv,rv,'o',color='SteelBlue',label='Chabrier - 1 src')
#pl.loglog(Msalpv,rv,'o',color='Teal',label='Salpeter - 1 src')
#pl.legend(loc='lower left')
xfit = np.logspace(10,12,100)
a,b=0.56,3.47*1e-6
yfit = b * xfit**a
pl.plot(xfit,yfit,'k--',label='Shen+03,z$\sim$0.1')

magv,magi,rmagv,rmagi,rmagb,muv,mui,rmuv,rmui,rv,ri,lumv,lumi,lumb = np.loadtxt('/data/ljo31/Lens/LensParams/PhotCat_2src.txt').T
ii=np.isfinite(lumi)
magv,magi,rmagv,rmagi,rmagb,muv,mui,rmuv,rmui,rv,ri,lumv,lumi,lumb = np.loadtxt('/data/ljo31/Lens/LensParams/PhotCat_2src.txt')[ii].T
names = ['J0913','J1125','J1144','J1323','J1347','J1446','J1605','J1606']
cols=['CornflowerBlue','Crimson','SandyBrown','PaleVioletRed','LightSeaGreen','LightSlateGray','Gold','DarkTurquoise']
Lv,Lb = 10**lumv, 10**lumb
ML_chabv,ML_chabb = 1.54, 2.15
ML_salpv, ML_salpb = 2.79, 3.90
Mchabb, Mchabv, Msalpb, Msalpv = Lb*ML_chabb, Lv*ML_chabv, Lb*ML_salpb, Lv*ML_salpv
#pl.figure()
for i in range(len(Lv)):
    #print names[i],cols[i]
    #pl.scatter(Mchabv[i],rv[i],marker='h',color=str(cols[i]),s=100)
    pl.scatter(Msalpv[i],rv[i],marker='*',color=str(cols[i]),s=100)
#pl.loglog(Mchabv,rv,'s',color='SteelBlue',label='Chabrier - 2 src')
#pl.loglog(Msalpv,rv,'s',color='Teal',label='Salpeter - 2 src')
#pl.legend(loc='lower right')
pl.xlabel('$\log(M/M_{\odot})$')
pl.ylabel('$\log(R_e/kpc)$')
#pl.text(3.45e11,1.5,'circle=1 src, Chabrier')
#pl.text(3.45e11,1.1,'square=1 src, Salpeter')
#pl.text(3.45e11,0.7,'hex=2 src, Chabrier')
#pl.text(3.45e11,0.45,'star=2 src, Salpeter')
vdWfit = 10**0.42 * (xfit/(5e10))**0.71
vdWfit2 = 10**0.60 * (xfit/(5e10))**0.75
pl.plot(xfit,vdWfit,'k:',label='ven der Wel+14, z=0.75')
pl.plot(xfit,vdWfit2,'k-.',label='van der Wel+14, z=0.25')
pl.legend(loc='lower right')
