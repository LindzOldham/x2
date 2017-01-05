import numpy,pylab,time
import MassModels as MM
import indexTricks as iT
from imageSim import convolve,SBModels
from scipy import ndimage,signal

xl = 30.1
yl = 30.1
b = 3.2
q = 0.7
pa = 0.
re = 8.1
n = 3.3

Y,X = iT.coords((61,61))

lens = MM.Sersic('',{'x':xl,'y':yl,'b':b,'q':q,'pa':pa,'re':re,'n':n})
t = time.time()
dx,dy = lens.deflections(X,Y)
print time.time()-t

def clens(x,y,b,q,pa,re,n,x0,y0):
    from math import cos,sin,pi
    from scipy.special import gamma,gammainc
    gal = SBModels.Sersic('',{'x':0.,'y':0.,'q':q,'pa':0.,'re':re,'n':n})

    Y,X = iT.coords(x0.shape)

    A = int(abs(x0-x).max())
    B = int(abs(y0-y).max())
    print B,A
    Y,X = iT.coords((B,A))

    k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
    amp = b*b*k**(2*n)/(numpy.exp(k)*2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
    kappa = amp*gal.pixeval(X,Y)


    X -= X.mean()
    Y -= Y.mean()

    R2 = X**2+Y**2
    R2[R2==0] = 1e-13

    dx,kfft = convolve.convolve(X/R2,kappa/pi)
    pylab.imshow(dx)
    pylab.show()

    dy = convolve.convolve(Y/R2,kfft,False)[0]
    return dx,dy

t = time.time()
Y,X = iT.overSample((61,61),4)
ddx,ddy = clens(xl,yl,b,q,pa,re,n,X,Y)
ddx = iT.resample(ddx,4)/16.
print time.time()-t

pylab.imshow((dx-ddx)/dx,origin='lower',interpolation='nearest')#,vmin=-2.,vmax=2.)
pylab.colorbar()
pylab.figure()
pylab.imshow(dx,origin='lower',interpolation='nearest')
pylab.colorbar()
pylab.figure()
pylab.imshow(ddx,origin='lower',interpolation='nearest')
pylab.colorbar()



pylab.show()


