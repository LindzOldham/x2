import numpy,pylab
import MassModels as MM
import indexTricks as iT
from imageSim import convolve
from scipy import ndimage,signal

xl = 30.3
yl = 30.3
b = 6.2
q = 1.
pa = 0.
eta = 0.5

Y,X = iT.coords((61,61))

lens = MM.PowerLaw('',{'x':xl,'y':yl,'b':b,'q':q,'pa':pa,'eta':eta})
dx,dy = lens.deflections(X,Y)

def clens(x,y,b,q,pa,eta,x0,y0):
    from math import cos,sin,pi
    x = x0-x
    y = y0-y
    theta = pa*pi/180.
    C = cos(theta)
    S = sin(theta)
    X = x*C+y*S
    Y = y*C-x*S

    kappa = 0.5*(2-eta)*b**eta/(q*X**2+Y**2/q)**(eta/2.)

    x0 -= x0.mean()
    y0 -= y0.mean()
    R2 = x0**2+y0**2
    R2[R2==0] = 1e-13

    ""
    pylab.imshow(x0/R2,origin='lower',interpolation='nearest')
    pylab.colorbar()
    pylab.figure()
    pylab.imshow(kappa,origin='lower',interpolation='nearest')
    pylab.colorbar()    
    pylab.show()
    ""

    dx,kfft = convolve.convolve(x0/R2,kappa/pi)
    dx = signal.convolve(x0/R2,kappa/pi,mode='same')
    dy = convolve.convolve(y0/R2,kfft,False)[0]
    return dx,dy

#Y,X = iT.overSample((61,61),4)
ddx,ddy = clens(xl,yl,b,q,pa,eta,X,Y)
#ddx = iT.resample(ddx,4)/16.

pylab.imshow((dx-ddx)/dx,origin='lower',interpolation='nearest')
pylab.colorbar()
pylab.figure()
pylab.imshow(dx,origin='lower',interpolation='nearest')
pylab.colorbar()
pylab.figure()
pylab.imshow(ddx,origin='lower',interpolation='nearest')
pylab.colorbar()



pylab.show()


