import numpy as np, pylab as pl, pyfits as py
from linslens import EELsImages as E
from imageSim import SBObjects, convolve
import indexTricks as iT
'''
plot,square=True,75
image = py.open('/data/ljo31/Lens/sharpdata/J1131_nirc2_w_Kp_6x6.fits')[0].data.copy()
# wide camera has 0.04 arcsec/pixel so a 2" aperture is now 50 pixels!!! And 75 pixels is 3 arcsec, perfecty.

K_2mass = 12.927
Kcorr = 1.87
kern = SBObjects.Gauss('kernel',{'x':0,'y':0,'sigma':50./2.322,'q':1,'pa':0,'amp':1})
xk,yk = iT.coords((121,121))-60.
kernel = kern.pixeval(xk,yk)
kernel = kernel/np.sum(kernel)
kernelc = convolve.convolve(image,kernel)[1]
blur = convolve.convolve(image,kernelc,False)[0]
if plot:
    pl.figure()
    pl.imshow(blur,interpolation='nearest',origin='lower')
    pl.colorbar()
    pl.figure()
    pl.imshow(image,interpolation='nearest',origin='lower',vmin=0,vmax=100)
    pl.colorbar()
y,x=iT.coords(image.shape)-square
R=np.sqrt(x**2.+y**2.)
flux = np.sum(blur[np.where(R<square)])
logged = -2.5*np.log10(flux)
mag = K_2mass + Kcorr
print K_2mass
ZP = mag-logged
print ZP
'''
# ZP = 28.117
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


# also J1619
image = py.open('/data/ljo31/Lens/J1619/J1619_nirc2_n_Kp_6x6.fits')[0].data.copy()
Kcorr = 1.87
# K_2MASS(Vega)
K_2mass = dict([('J0837',15.074),('J0901',15.202),('J0913',0),('J1125',14.945),('J1144',15.107),('J1218',14.841),('J1248',14.670),('J1323',14.980),('J1347',15.341),('J1446',0),('J1605',15.216),('J1606',15.223),('J1619',14.853),('J2228',14.987)])
# uncertainties on the above (just in case!)
dK_2mass = dict([('J0837',0.115),('J0901',0.111),('J0913',0),('J1125',0.107),('J1144',0.110),('J1218',0.135),('J1248',0.098),('J1323',0.118),('J1347',0.195),('J1446',0.164),('J1605',0),('J1606',0.142),('J1619',0.094),('J2228',0.148)])
ZP = GetZP(image,kernel,300,300,'J1323',plot=True)
print ZP


