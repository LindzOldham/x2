import numpy as np, pylab as pl, pyfits as py
from imageSim import SBObjects, convolve
import indexTricks as iT

# start with all narrow-camera objects!
ZPs = []
square,ap = 350,350

# K_2MASS(Vega)
K_2mass = dict([('J0837',15.074),('J0901',15.202),('J0913',0),('J1125',14.945),('J1144',15.107),('J1218',14.841),('J1248',14.670),('J1323',14.980),('J1347',15.341),('J1446',0),('J1605',15.216),('J1606',15.223),('J1619',14.853),('J2228',14.987)])
# uncertainties on the above (just in case!)
dK_2mass = dict([('J0837',0.115),('J0901',0.111),('J0913',0),('J1125',0.107),('J1144',0.110),('J1218',0.135),('J1248',0.098),('J1323',0.118),('J1347',0.195),('J1446',0.164),('J1605',0),('J1606',0.142),('J1619',0.094),('J2228',0.148)])
# entries are zero where system wasn't observed in 2MASS
Kcorr = 1.87

xk,yk = iT.coords((501,501))-250.
kern = SBObjects.Gauss('kernel',{'x':0,'y':0,'sigma':200./2.322,'q':1,'pa':0,'amp':1})
kernel = kern.pixeval(xk,yk)
kernel = kernel/np.sum(kernel)
def GetZP(image,kernel,square,ap,name,plot=False):
    kernelc = convolve.convolve(image,kernel)[1]
    blur = convolve.convolve(image,kernelc,False)[0]
    if plot:
       pl.figure()
       pl.imshow(blur,interpolation='nearest',origin='lower')
       pl.colorbar()
       pl.figure()
       pl.imshow(image,interpolation='nearest',origin='lower',vmin=0,vmax=4)#np.amax(image)*0.25)
       pl.colorbar()
       print np.sum(blur)/ np.sum(image) 
    y,x=iT.coords(image.shape)-square
    R=np.sqrt(x**2.+y**2.)
    flux = np.sum(blur[np.where(R<ap)])
    logged = -2.5*np.log10(flux)
    mag = K_2mass[name] + Kcorr
    print K_2mass[name]
    ZP = mag-logged
    return ZP#,blur

# J1125
s=250
image = py.open('/data/ljo31/Lens/J1125/Kp_J1125_nirc2_n.fits')[0].data.copy()[775-s:775+s,775-s:775+s]
image[np.where(np.isnan(image)==True)] = 0.0
y,x=iT.coords(image.shape)
r=np.sqrt((x-s)**2. + (y-s)**2.)
image[(image<-2) & (r>200)] = 0.
image[(image>2) & (r>200)] = 0.
pl.figure()
pl.imshow(image,interpolation='nearest',origin='lower',vmin=0,vmax=1)
im = py.open('/data/ljo31/Lens/J1125/Kp_J1125_nirc2_n.fits')[0].data.copy()[775-s:775+s,775-s:775+s]
im[np.where(np.isnan(im)==True)] = 0.0
pl.figure()
pl.imshow(image-im,interpolation='nearest',origin='lower')
kk = np.where(abs(image-im)!=0)
print image[kk]-im[kk]
ZP = GetZP(image,kernel,s,s,'J1125',plot=False)
ZPs.append(('J1125',ZP))
print dict(ZPs)

from linslens import EELsKeckLensModels as L
result = np.load('/data/ljo31/Lens/J1125/Kp_211_0')
hstresult = np.load('/data/ljo31/Lens/LensModels/J1125_211')
model = L.EELs(result,hstresult,name='J1125')
model.MakeDict()
model.BuildLenses()
model.BuildGalaxies()
model.BuildSources()
model.EasyAddImages()
model.GetFits(plotresid=False)
fits, gals = model.fits, model.gals
mk1,mk2 = gals[0].getMag(fits[0],ZP), gals[1].getMag(fits[1],ZP)
Fk = 10**(0.4*(ZP-mk1)) + 10**(0.4*(ZP-mk2))
mag_k = -2.5*np.log10(Fk) + ZP
print s,mag_k
