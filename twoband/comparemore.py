import pylab as pl, pyfits as py, numpy as np

phot = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
names = phot['name']

re,rel,reu = phot['Re v'], phot['Re v lo'], phot['Re v hi']
dre = np.mean((rel,reu),axis=0)
mu,mul,muu = phot['mu v'], phot['mu v lo'], phot['mu v hi']
dmu = np.mean((mul,muu),axis=0)


tab1 = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
tab2 = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge.fits')[1].data

re1,mag1,mu1,names1 = tab1['re i'], tab1['mag i'], tab1['mu i'],tab1['name']
re2,mag2,mu2,names2 = tab2['re i'], tab2['mag i'], tab2['mu i'],tab2['name']
re3,mag3,mu3 = re2*0,mag2*0,mu2*0

for name in names2:
    ii = np.where(names1==name)
    mag3[np.where(names2==name)] = mag1[ii]
    mu3[np.where(names2==name)] = mu1[ii]
    re3[np.where(names2==name)] = re1[ii]

for name in names2:
    ii=np.where(names2==name)
    print name, '%.2f'%re2[ii], '%.2f'%re3[ii], '%.2f'%(re2[ii]-re3[ii])

'''pl.figure()
xline=np.linspace(15,30,20)
pl.scatter(mag2,mag3,s=30,color='SteelBlue')
pl.plot(xline,xline)
pl.figure()
pl.scatter(mu2,mu3,s=30,color='SteelBlue')'''

# brighter things get fainter, fainter things get lighter -- things get less extreme?


cat1 = np.load('/data/ljo31/Lens/LensParams/Structure_1src.npy')[()][0]
cat2 = np.load('/data/ljo31/Lens/LensParams/Structure_1src_huge.npy')[()][0]

n,eta,b,re1,re2 = [],[],[],[],[]
n1,n2 = [],[]
for name in names2:
    ii=np.where(names1==name)
    jj = np.where(names2==name)
    n.append([cat1[name]['Source 1 n'],cat2[name]['Source 1 n']])
    eta.append([cat1[name]['Lens 1 eta'],cat2[name]['Lens 1 eta']])
    b.append([cat1[name]['Lens 1 b'],cat2[name]['Lens 1 b']])
    ## galaxy light properties
    re1.append([cat1[name]['Galaxy 1 re'],cat2[name]['Galaxy 1 re']])
    re2.append([cat1[name]['Galaxy 2 re'],cat2[name]['Galaxy 2 re']])
    n1.append([cat1[name]['Galaxy 1 n'],cat2[name]['Galaxy 1 n']])
    n2.append([cat1[name]['Galaxy 2 n'],cat2[name]['Galaxy 2 n']])


n,eta,b,re1,re2,n1,n2 = np.array(n),np.array(eta),np.array(b),np.array(re1),np.array(re2),np.array(n1),np.array(n2)

def scat(x,y):
    pl.scatter(x,y,s=30,color='SteelBlue')
    pl.axhline(0,color='SteelBlue')

pl.figure()
scat(n[:,1],n[:,1]-n[:,0])
pl.xlabel('source Sersic index (new)')
pl.ylabel('new - old')

pl.figure()
scat(eta[:,1],eta[:,1]-eta[:,0])
pl.xlabel('lens power law index (new)')
pl.ylabel('lens power law index (old)')
pl.ylabel('new - old')

pl.figure()
scat(b[:,1],b[:,1]-b[:,0])
pl.xlabel('Einstein radius (new)')
pl.ylabel('Einstein radius (old)')
pl.ylabel('new - old')

pl.figure()
scat(re1[:,1],re1[:,1]-re1[:,0])
pl.xlabel('galaxy 1 $r_e$ (new)')
pl.ylabel('galaxy 1 $r_e$ (old)')
pl.ylabel('new - old')

pl.figure()
scat(re2[:,1],re2[:,1]-re2[:,0])
pl.xlabel('galaxy 2 $r_e$ (new)')
pl.ylabel('galaxy 2 $r_e$ (old)')
pl.ylabel('new - old')

pl.figure()
scat(n1[:,1],n1[:,1]-n1[:,0])
pl.xlabel('galaxy 1 sersic index (new)')
pl.ylabel('galaxy 1 sersic index (old)')
pl.ylabel('new - old')

pl.figure()
scat(n2[:,1],n2[:,1]-n2[:,0])
pl.xlabel('galaxy 2 sersic index (new)')
pl.ylabel('galaxy 2 sersic index (old)')
pl.ylabel('new - old')

