import numpy as np, pylab as pl, pyfits as py
import cPickle
from EllipsePlot import *

xx,yy,zz,dxx,dyy,dzz = np.load('/data/ljo31b/EELs/esi/kinematics/FP_EELs.npy').T
xxc,yyc,zzc,dxxc,dyyc,dzzc = np.load('/data/ljo31b/EELs/esi/kinematics/FP_Coma.npy').T
dxxc,dyyc,dzzc = dxxc**0.5,dyyc**0.5,dzzc**0.5
rho = 0.99


d,u,l = np.load('/data/ljo31b/EELs/FP/inference/FP_Coma_results.npy').T
alphac,ac,bc = d[0]['alpha'],d[0]['a'],d[0]['b']
alphac_l, alphac_h = l[0]['alpha'],u[0]['alpha']
d,u,l = np.load('/data/ljo31b/EELs/FP/inference/FP_results.npy').T
alpha,a,b = d[0]['alpha'],d[0]['a'],d[0]['b']
alpha_l, alpha_h = l[0]['alpha'],u[0]['alpha']


# plot eels
line = np.linspace(np.min(zzc)-0.1,np.max(zzc)+0.2,10)
pl.figure(figsize=(15,7))
pl.subplot(121)
pl.plot(line-alpha,line,color='Gray')
pl.fill_between(line-alpha,line-alpha_l,line+alpha_h,color='Gainsboro',alpha=0.5)
pl.fill_between(line,line-alpha,line-alpha+alpha_h,color='Gainsboro',alpha=0.5)
pl.scatter(a*xx+b*yy,zz,color='Gray',label='EELs')
pl.ylim([line[0],line[-1]])
pl.xlim([7.4,9.5])
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%(a)+r'$\log\sigma$ + $'+'%.2f'%(b)+r'$\mu$')
pl.title('EELs')
## plot coma
pl.subplot(122)
pl.plot(line,line-alphac,color='Gray')
pl.fill_between(line,line-alphac,line-alphac-alphac_l,color='Gainsboro',alpha=0.5)
pl.fill_between(line,line-alphac,line-alphac+alphac_h,color='Gainsboro',alpha=0.5)
pl.scatter(ac*xxc+bc*yyc,zzc,color='Gray',label='Coma')
pl.title('Coma')
pl.ylim([line[0],line[-1]])
pl.xlim([7.4,9.5])
pl.ylabel(r'$\log R_e$')
pl.xlabel('%.2f'%(ac)+r'$\log\sigma$ + $'+'%.2f'%(bc)+r'$\mu$')
#pl.legend(loc='upper left')


'''
A = np.array([[0,0,1],[ac,bc,0],[0,0,0]])
sigmaXlist_coma, sigmaYlist_coma, rhoXYlist_coma = find_sigmaz(xxc,yyc,zzc,dxxc,dyyc,dzzc,rho*np.ones(yyc.size),A,a,b)
A = np.array([[0,0,1],[a,b,0],[0,0,0]])
sigmaXlist, sigmaYlist, rhoXYlist = find_sigmaz(xx,yy,zz,dxx,dyy,dzz,rho*np.ones(yy.size),A,a,b)
#plot_ellipses(zz,a*xx+b*yy,sigmaXlist, sigmaYlist, rhoXYlist,'Crimson')
#plot_ellipses(zzc,ac*xxc+bc*yyc,sigmaXlist_coma, sigmaYlist_coma, rhoXYlist_coma,'Gray')

# plot the face-on projection
pl.figure()
pl.scatter(ac*xx + bc*yy + np.sqrt(ac**2.+bc**2.)*zz, -bc*xx + ac*yy, color='Crimson',label='EELs')
pl.scatter(ac*xxc + bc*yyc + np.sqrt(ac**2.+bc**2.)*zzc, -bc*xxc + ac*yyc, color='Gray',label='Coma')
pl.xlabel(r'$\alpha\log\sigma + \beta\mu + (\alpha^2 + \beta^2)^0.5\log R_e$')
pl.ylabel(r'$-\beta\log\sigma + \alpha\mu$')
pl.axis([7,10.5,17.5,23.4])
pl.legend(loc='upper left')

A = np.array([[ac, bc,np.sqrt(ac**2+bc**2)],[-bc,ac,0],[0,0,0]])
sigmaXlist_coma, sigmaYlist_coma, rhoXYlist_coma = find_sigmaz(xxc,yyc,zzc,dxxc,dyyc,dzzc,rho*np.ones(yyc.size),A,a,b)
#plot_ellipses(ac*xxc + bc*yyc + np.sqrt(ac**2.+bc**2.)*zzc, -bc*xxc + ac*yyc, sigmaXlist_coma, sigmaYlist_coma, rhoXYlist_coma,'Gray')
A = np.array([[a, b,np.sqrt(a**2+b**2)],[-b,a,0],[0,0,0]])
sigmaXlist, sigmaYlist, rhoXYlist = find_sigmaz(xx,yy,zz,dxx,dyy,dzz,rho*np.ones(yy.size),A,a,b)
#plot_ellipses(a*xx + b*yy + np.sqrt(a**2.+b**2.)*zz, -b*xx + a*yy, sigmaXlist_macs, sigmaYlist_macs, rhoXYlist_macs,'Crimson')
'''
