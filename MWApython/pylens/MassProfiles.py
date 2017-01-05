from ndinterp import ndInterp
from math import pi

class MassProfile:
    """
    A generic mass model class, including a method to calculate deflection
        angles for power law models. This probably needs to be re-worked.
    """
    def align_coords(self,xin,yin,offset=True,revert=False):
        from math import cos,sin,pi
        if offset:
            theta = self.theta-pi/2.
        else:
            theta = self.theta
        ctheta = cos(theta)
        stheta = sin(theta)
        if revert:
            X = xin*ctheta-yin*stheta
            Y = yin*ctheta+xin*stheta
            x = X+self.x
            y = Y+self.y
            return x,y
        X = xin-self.x
        Y = yin-self.y
        x = X*ctheta+Y*stheta
        y = Y*ctheta-X*stheta
        return x,y


class PowerLaw(MassProfile):
    """
    A subclass for power-law mass models. The `power-law' aspect doesn't
        currently work, but it does work for SIE models.
    """
    def __init__(self,b=None,eta=1.,pa=None,q=None,x=None,y=None,load=False):
        self.b = b
        self.eta = eta
        self.pa = pa
        self.q = q
        self.x = x
        self.y = y
        self.nsteps = 1e3
        if load:
            import cPickle
            f = open('powerlaw.alphax','rb')
            self.xmodel = cPickle.load(f)
            f.close()
            f = open('powerlaw.alphay','rb')
            self.ymodel = cPickle.load(f)
            f.close()
        else:
            self.xmodel=None
            self.ymodel=None


    def kappa(self,rho):
        return 0.5*rho**(self.eta-2.)

    def deflections(self,xin,yin):
        from numpy import ones,arctan as atan, arctanh as atanh
        from math import cos,sin,pi
        from numpy import arcsin as asin,arcsinh as asinh
        x,y = self.align_coords(xin,yin)
        q = self.q
        if q==1.:
            q = 1.-1e-7  # Avoid divide-by-zero errors
        eps = (1.-q**2)**0.5
        if self.eta==1.:
            # SIE models
            r = (x**2+y**2)**0.5
            r[r==0.] = 1e-9
            xout = self.b*asinh(eps*x/q/r)*q**0.5/eps
            yout = self.b*asin(eps*y/r)*q**0.5/eps
        else:
            from powerlaw import powerlawdeflections as pld
            b,eta = self.b,self.eta
            s = 1e-7
            if x.ndim>1:
                yout,xout = pld(-1*y.ravel(),x.ravel(),b,eta,s,q)
                xout,yout = xout.reshape(x.shape),-1*yout.reshape(y.shape)
            else:
                yout,xout = pld(-1*y,x,b,eta,s,q)
                yout = -1*yout
        theta = -(self.theta - pi/2.)
        ctheta = cos(theta)
        stheta = sin(theta)
        x = xout*ctheta+yout*stheta
        y = yout*ctheta-xout*stheta
        return x,y

    def caustic(self):
        if self.eta!=1:
            return None,None
        from numpy import ones,arctan as atan, arctanh as atanh
        from numpy import cos,sin,pi,linspace
        from numpy import arcsin as asin,arcsinh as asinh
        q = self.q
        if q==1.:
            q = 1.-1e-7  # Avoid divide-by-zero errors
        eps = (1.-q**2)**0.5
        theta = linspace(0,2*pi,5000)
        ctheta = cos(theta)
        stheta = sin(theta)
        delta = (ctheta**2+q**2*stheta**2)**0.5
        xout = ctheta*q**0.5/delta - asinh(eps*ctheta/q)*q**0.5/eps
        yout = stheta*q**0.5/delta - asin(eps*stheta)*q**0.5/eps
        xout,yout = xout*self.b,yout*self.b
        theta = -(self.theta - pi/2.)
        ctheta = cos(theta)
        stheta = sin(theta)
        x = xout*ctheta+yout*stheta
        y = yout*ctheta-xout*stheta
        return x+self.x,y+self.y

    def getKappa(self,xin,yin):
        x,y = self.align_coords(xin,yin,offset=False)
        R = (self.q*x**2+y**2/self.q)**0.5
        return 0.5*(2.-self.eta)*(self.b/R)**self.eta


class SIS(MassProfile):

    def __init__(self,x=None,y=None,b=0.):
        self.x = x
        self.y = y
        self.b = b

    def deflections(self,x,y):
        dx = x-self.x
        dy = y-self.y
        amp = (dx**2+dy**2)**0.5
        return self.b*dx/amp,self.b*dy/amp


class ExtShear(MassProfile):

    def __init__(self,x=None,y=None,b=0.,theta=0.,pa=None):
        self.x = x
        self.y = y
        self.b = b
        if pa is None:
            self.theta = theta
        else:
            self.pa = pa

    def deflections(self,x,y):
        from math import sin,cos,pi
        x = x-self.x
        y = y-self.y
        theta = self.theta #- pi/2
        s = sin(2*theta)
        c = cos(2*theta)

        # From Kormann B1422 paper
        alpha_x = -self.b*(x*c+y*s)
        alpha_y = -self.b*(x*s-y*c)

        return alpha_x,alpha_y


class PointSource(MassProfile):
    def __init__(self,x=None,y=None,b=0.):
        self.x = x
        self.y = y

    def deflections(self,x,y):
        from math import pi 
        x = x-self.x
        y = y-self.y
        r2 = x**2+y**2
        b2 = self.b**2
        alpha_x = b2*x/r2
        alpha_y = b2*y/r2

        return alpha_x,alpha_y


class Sersic(MassProfile):
    def __init__(self,b=None,n=None,pa=None,q=None,x=None,y=None,re=None):
        self.b = b
        self.n = n
        self.pa = pa
        self.q = q
        self.x = x
        self.y = y
        self.re = re

    def deflections(self,xin,yin):
        from numpy import ones,arctan as atan, arctanh as atanh
        from math import cos,sin,pi,exp
        from numpy import arcsin as asin,arcsinh as asinh
        from scipy.special import gammainc,gamma
        x,y = self.align_coords(xin,yin)
        q = self.q
        if q==1.:
            q = 1.-1e-7  # Avoid divide-by-zero errors
        from sersic import sersicdeflections as sd
        b,n,re = self.b,self.n,self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        b = b/q**0.5
        re = re/q**0.5
        amp = b*b*k**(2*n)/(2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        if x.ndim>1:
            yout,xout = sd(-1*y.ravel(),x.ravel(),amp,re,n,q)
            xout,yout = xout.reshape(x.shape),-1*yout.reshape(y.shape)
        else:
            yout,xout = sd(-1*y,x,amp,re,n,q)
            yout = -1*yout
        theta = -(self.theta - pi/2.)
        ctheta = cos(theta)
        stheta = sin(theta)
        x = xout*ctheta+yout*stheta
        y = yout*ctheta-xout*stheta
        return x,y

    def getMass(self,sigCrit=1.):
        import numpy
        from math import pi
        from scipy.special import gammainc,gamma

        b,n,re = self.b,self.n,self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        amp = b*b*k**(2*n)/(2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        return 2*pi*sigCrit*re**2*amp*gamma(2*n)*n/k**(2*n)
        ###
        amp = b*b*k**(2*n)/(numpy.exp(k)*2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        return 2*pi*sigCrit*re**2*amp*gamma(2*n)*n/(numpy.exp(k)*k**(2*n))

    def getbFromMass(self,mass,sigCrit):
        from scipy import interpolate
        from scipy.special import gammainc,gamma
        import numpy

        n,re = self.n,self.re
        b = numpy.logspace(-3,2,501)*re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
#        amp = b*b*k**(2*n)/(numpy.exp(k)*2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
#        m = 2*pi*sigCrit*re**2*amp*gamma(2*n)*n/(numpy.exp(k)*k**(2*n))
        amp = b*b*k**(2*n)/(2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        m = 2*pi*sigCrit*re**2*amp*gamma(2*n)*n/k**(2*n)
        model = interpolate.splrep(m,b)
        return interpolate.splev(mass,model)

    def setbFromMass(self,mass,sigCrit):
        self.b = self.getbFromMass(mass,sigCrit)

    def getKappa(self,xin,yin):
        import numpy
        from scipy.special import gamma,gammainc
        x,y = self.align_coords(xin,yin,offset=False)
        R = (self.q*x**2+y**2/self.q)**0.5

        b = self.b
        n = self.n
        re = self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)

        amp = b*b*k**(2*n)/(numpy.exp(k)*2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        return amp*numpy.exp(-k*((R/re)**(1./n)-1.))


class SersicG(MassProfile,Sersic):
    def __init__(self,b=None,n=None,pa=None,q=None,x=None,y=None,re=None):
        import numpy
        import pylens
        self.b = b
        self.n = n
        self.pa = pa
        self.q = q
        self.x = x
        self.y = y
        self.re = re
        path = pylens.__file__.split('pylens.py')[0]
        self.xmod,self.ymod = numpy.load('%s/serModelsHDR.dat'%path)

    def deflections(self,xin,yin):
        import numpy
        from math import cos,sin,pi
        from scipy.special import gammainc
        x,y = self.align_coords(xin,yin)
        q = self.q
        if q==1.:
            q = 1.-1e-7  # Avoid divide-by-zero errors
        b,n,re = self.b,self.n,self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        amp = (b/re)**2/gammainc(2*n,k*(b/re)**(1/n))
        r0 = re/q**0.5
        o = numpy.ones(x.size).astype(x.dtype)
        eval = numpy.array([abs(x).ravel()/r0,abs(y).ravel()/r0,n*o,q*o]).T
        xout = numpy.sign(x)*(amp*self.xmod.eval(eval)*r0).reshape(x.shape)
        yout = numpy.sign(y)*(amp*self.ymod.eval(eval)*r0).reshape(y.shape)
        theta = -(self.theta - pi/2.)
        ctheta = cos(theta)
        stheta = sin(theta)
        x = xout*ctheta+yout*stheta
        y = yout*ctheta-xout*stheta
        return x,y

"""
    def getMass(self,sigCrit=1.):
        from math import pi
        from scipy.special import gammainc,gamma

        b,n,re = self.b,self.n,self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        amp = b*b*k**(2*n)/(2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        return 2*pi*sigCrit*re**2*amp*gamma(2*n)*n/k**(2*n)

    def getbFromMass(self,mass,sigCrit):
        from scipy import interpolate
        from scipy.special import gammainc,gamma
        import numpy

        n,re = self.n,self.re
        b = numpy.logspace(-3,1,401)*re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        amp = b*b*k**(2*n)/(2*n*re**2*(gamma(2*n)*gammainc(2*n,k*(b/re)**(1/n))))
        m = 2*pi*sigCrit*re**2*amp*gamma(2*n)*n/k**(2*n)
        model = interpolate.splrep(m,b)
        return interpolate.splev(mass,model)

    def setbFromMass(self,mass,sigCrit):
        self.b = self.getbFromMass(mass,sigCrit)
"""


class sNFW(MassProfile):
    def __init__(self,b=None,rs=None,x=None,y=None):
        self.b = b
        self.rs = rs
        self.x = x
        self.y = y
        self.q = 1.
        self.pa = 0.
        self.theta = 0.

    def deflections(self,xin,yin):
        from numpy import arctanh,arctan,arctan2,log,sin,cos

        #x,y = self.align_coords(xin,yin)
        x = xin-self.x
        y = yin-self.y
        b,rs = self.b,self.rs
        X = b/rs
        if X<1.:
            amp = X**2/(8*arctanh(((1-X)/(1+X))**0.5)/(1-X**2)**0.5+4*log(X/2.))
        elif X==1:
            amp = 0.25/(1.+log(0.5))
        else:
            amp = X**2/(8*arctan(((X-1)/(1+X))**0.5)/(X**2-1)**0.5+4*log(X/2.))

        r2 = (x**2+y**2)/rs**2
        r = r2**0.5
        F = r*0.
        F[r<1.] = arctanh((1-r2[r<1.])**0.5)/(1-r2[r<1.])**0.5
        F[r==1.] = 1.
        F[r>1.] = arctan((r2[r>1.]-1.)**0.5)/(r2[r>1.]-1)**0.5

        dr = 4*amp*(log(r/2)+F)/r
        return dr*x/r,dr*y/r
        A = arctan2(y,x)
        return dr*cos(A),dr*sin(A)

    def getbFromMass(self,mass,rvir,sigCrit):
        from numpy import arctanh,arctan,arctan2,log,sin,cos,pi,logspace
        from scipy import interpolate

        rs = self.rs
        c = rvir/self.rs

        x = logspace(-4,1,501)
        A = x*0.
        C = x<1.
        X = x[C].copy()
        A[C] = X**2/(8*arctanh(((1-X)/(1+X))**0.5)/(1-X**2)**0.5+4*log(X/2.))
        C = x==1.
        A[C] = 0.25/(1.+log(0.5))
        C = x>1.
        X = x[C].copy()
        A[C] = X**2/(8*arctan(((X-1)/(1+X))**0.5)/(X**2-1)**0.5+4*log(X/2.))

        m = 4*pi*A*rs**2*(log(1+c)-c/(1+c))*sigCrit
        model = interpolate.splrep(m,x*rs)
        return interpolate.splev(mass,model)

    def setbFromMass(self,mass,rvir,sigCrit):
        self.b = self.getbFromMass(mass,rvir,sigCrit)

    def kappa(self,r):
        from numpy import arctanh,arctan,arctan2,log,sin,cos,pi,logspace
        x = self.b/self.rs

        if x<1.:
            norm = x**2/(4*arctanh(((1-x)/(1+x))**0.5)/(1-x**2)**0.5+2*log(x/2))
        elif x==1.:
            norm = 1./(2.+2*log(0.5))
        else:
            norm = x**2/(4*arctan(((x-1)/(x+1))**0.5)/(x**2-1)**0.5+2*log(x/2))

        x = r/self.rs
        A = x*0.
        C = x<1.
        X = x[C].copy()
        A[C] = (1.-2*arctanh(((1.-X)/(1.+X))**0.5)/(1-X**2)**0.5)/(X**2-1.)
        C = x==1.
        A[C] = 1./3
        C = x>1.
        X = x[C].copy()
        A[C] = (1.-2*arctan(((X-1.)/(1.+X))**0.5)/(X**2-1.)**0.5)/(X**2-1.)
        return norm*A


class eNFWp(MassProfile):
    def __init__(self,b=None,rs=None,x=None,y=None):
        self.b = b
        self.rs = rs
        self.x = x
        self.y = y
        self.q = 1.
        self.pa = 0.
        self.theta = 0.

    def deflectionsOLD(self,xin,yin):
        from numpy import arctanh,arctan,arctan2,log,sin,cos
        x,y = self.align_coords(xin,yin)
        
        b,rs,q = self.b,self.rs,self.q
        X = b/rs
        if X<1.:
            amp = X**2/(8*arctanh(((1-X)/(1+X))**0.5)/(1-X**2)**0.5+4*log(X/2.))
        elif X==1:
            amp = 0.25/(1.+log(0.5))
        else:
            amp = X**2/(8*arctan(((X-1)/(1+X))**0.5)/(X**2-1)**0.5+4*log(X/2.))

        r2 = (q*x**2+y**2/q)/rs**2
        r = r2**0.5
        F = r*0.
        F[r<1.] = arctanh((1-r2[r<1.])**0.5)/(1-r2[r<1.])**0.5
        F[r==1.] = 1.
        F[r>1.] = arctan((r2[r>1.]-1.)**0.5)/(r2[r>1.]-1)**0.5

        dr = 4*amp*rs*(log(r/2)+F)/r
        A = arctan2(y,x)
        return dr*cos(A)*q**0.5,dr*sin(A)/q**0.5

    def deflections(self,xin,yin):
        import numpy
        from numpy import arctanh,arctan,arctan2,arccosh,arccos,log,sin,cos
        x,y = self.align_coords(xin,yin,False)

        b,rs,q = self.b,self.rs,self.q
        X = b/rs
        if X<1.:
            amp = X**2/(8*arctanh(((1-X)/(1+X))**0.5)/(1-X**2)**0.5+4*log(X/2.))
        elif X==1:
            amp = 0.25/(1.+log(0.5))
        else:
            amp = X**2/(8*arctan(((X-1)/(1+X))**0.5)/(X**2-1)**0.5+4*log(X/2.))

        x2 = x**2
        y2 = y**2        
        r2 = (q*x2+y2/q)/rs**2
        r = r2**0.5
        F = r*0.

        arg = (1-r2[r<1.])**0.5
        F[r<1.] = arctanh(arg)/arg
        F[r==1.] = 1.
        arg = (r2[r>1.]-1.)**0.5
        F[r>1.] = arctan(arg)/arg

        dr = 4*amp*rs*(log(r/2)+F)/r
#        r = (x2+y2)**0.5
#        r = (r2*rs**2)**0.5
        r = r*rs
#        return dr*x*q/r,dr*y/r/q
        dx,dy = dr*x*q**0.5/r,dr*y/r/q**0.5
        ctheta = cos(self.theta)
        stheta = sin(self.theta)
        x = dx*ctheta-dy*stheta
        y = dy*ctheta+dx*stheta
        return x,y
        return dr*x*q**0.5/r,dr*y/r/q**0.5


class Jaffe(MassProfile):
    def __init__(self,b=None,rs=None,x=None,y=None):
        self.b = b
        self.rs = rs
        self.x = x
        self.y = y
        self.q = 1.
        self.pa = 0.
        self.theta = 0.

    def deflections(self,xin,yin):
        import numpy
        from numpy import arctanh,arctan,pi
        x = xin.copy()-self.x
        y = yin.copy()-self.y

        b,rs = self.b,self.rs

        X = b/rs
        # ignore a factor of 1/(2*rs)
        if X<1:
            amp = X/(pi-2*X*arctanh((1-X**2)**0.5)/(1-X**2)**0.5)
        elif X==1:
            amp = 1/(pi-2)
        else:
            amp = X/(pi-2*X*arctan((X**2-1)**0.5)/(X**2-1)**0.5)

        r2 = (x**2+y**2)/rs**2
        r = r2**0.5

        F = r*0.

        arg = (1-r2[r<1.])**0.5
        F[r<1.] = arctanh(arg)/arg
        F[r==1.] = 1.
        arg = (r2[r>1.]-1.)**0.5
        F[r>1.] = arctan(arg)/arg

        # ignore 2*rs again
        phi = amp*(pi-2*r*F)
        return phi*x/r,phi*y/r


class dPIE(MassProfile):
    def __init__(self,b=None,rs=None,x=None,y=None,q=None,pa=None):
        self.b = b
        self.rs = rs
        self.x = x
        self.y = y
        self.q = 1.
        self.pa = 0.
        self.theta = 0.

    def deflections(self,xin,yin,a=1e-12):
        from numpy import cos,sin
        x,y = self.align_coords(xin,yin,False)

        b,rs,q = self.b,self.rs,self.q

        r2 = (q*x**2+y**2/q)
        r = r2**0.5

#        f = (r/a)/(1+(1+r2/a**2)**0.5) - (r/rs)/(1+(1+r2/rs**2)**0.5)
        b2 = b**2
#        amp = -b2/(a-(a**2+b2)**0.5-rs+(b**2+rs**2)**0.5)
        f = 1.-(r/rs)/(1+(1+r2/rs**2)**0.5)
        amp = b2/(b+rs-(b2+rs**2)**0.5)
#        return amp*f*x/r,amp*f*y/r
        dx,dy = amp*f*x*q**0.5/r,amp*f*y/r/q**0.5
        ctheta = cos(self.theta)
        stheta = sin(self.theta)
        x = dx*ctheta-dy*stheta
        y = dy*ctheta+dx*stheta
        return x,y
        return amp*f*x*q**0.5/r,amp*f*yin/r/q**0.5


class sGNFW(MassProfile):
    # spherical cuspy halo model and n = 3
    def __init__(self,b=None,rs=None,eta=None,x=None,y=None):
        self.b = b
        self.rs = rs
        self.eta = eta
        self.x = x
        self.y = y
        self.pa = 0.
        self.theta = 0.
        self.ks = None

    def deflections(self,xin,yin):
        import numpy
        from scipy import interpolate
        from scipy.special import psi,hyp2f1,beta
        dx = xin-self.x
        dy = yin-self.y
        r = (dx**2+dy**2)**0.5

        G = self.eta
        R0 = self.rs
        B = self.b/R0

        xx = numpy.logspace(-6,0,61)*B
        xx2p1 = 1.+xx**2
        # Drop factors of 2 (from Beta term) and 1/2 (missing from Keeton)
        kappa = hyp2f1(1.,G/2.,1.5,1./xx2p1)/xx2p1
        c = numpy.isfinite(kappa)
        kappa = kappa[c]
        xx = xx[c]
        model = interpolate.splrep(xx,kappa*xx)
        ks = B**2/(2*interpolate.splint(xx[0],B,model))
        self.ks = ks

        x = r/R0
        x2 = x**2
        A = numpy.log(1.+x2)/x

        # First x>1
        c = x>=1.
        X = x[c]
        X2 = x2[c]
        if c.sum()>0:
            A[c] += (-self.Gfunc(G/2.,1.5,1./(1.+X2))+psi(1.5)-psi((3-G)/2.))/X

        # Then x<1
        c = x<1.
        X = x[c]
        X2 = x2[c]
        # deal with NFW case (g==1) for x<1
        if G==1 and c.sum()>0:
            G = 0.99
            tmp1 = A[c] -(self.Gfunc(G/2.,(G-1.)/2.,X2/(1.+X2))/X+(X**(2-G))*((1+X2)**((G-3)/2.))*beta((G-3)/2.,1.5)*hyp2f1(1.5,(3-G)/2.,(5-G)/2.,X2/(1+X2)))
            G = 1.01
            tmp2 = A[c] -(self.Gfunc(G/2.,(G-1.)/2.,X2/(1.+X2))/X+(X**(2-G))*((1+X2)**((G-3)/2.))*beta((G-3)/2.,1.5)*hyp2f1(1.5,(3-G)/2.,(5-G)/2.,X2/(1+X2)))
            A[c] = 10**(numpy.log10(tmp1)/2+numpy.log10(tmp2)/2.)
        elif c.sum()>0:
            A[c] += -(self.Gfunc(G/2.,(G-1.)/2.,X2/(1.+X2))/X+(X**(2-G))*((1+X2)**((G-3)/2.))*beta((G-3)/2.,1.5)*hyp2f1(1.5,(3-G)/2.,(5-G)/2.,X2/(1+X2)))

        deflection = ks*R0*A
        return deflection*dx/r,deflection*dy/r

    def Gfunc(self,b,c,z,nmax=100):
        import numpy
        if z.size==0:
            return
        if b==0:
            return z*0.
        top = b
        bot = c
        result = b*z/c
        indx = numpy.arange(z.size)
        for i in range(1,3):
            top *= b+i
            bot *= c+i
            n = i+1.
            term = (top/bot)*(z**n)/n
            result += term
        indx = indx[abs(term/result)>0.0001]
        z = z[indx]
        for i in range(3,nmax):
            top *= b+i
            bot *= c+i
            n = i+1.
            term = (top/bot)*(z**n)/n
            cond = abs(term/result[indx])>0.0001
            result[indx] += term
            if cond.sum()==0:
                break
            z = z[cond]
            indx = indx[cond]
        return result

    def getMass(self,R):
        import numpy
        from scipy import interpolate
        from scipy.special import psi,hyp2f1,beta

        G = self.eta
        R0 = self.rs
        B = self.b/R0
        R = R/R0

        if R<B:
            xx = numpy.logspace(-6,0,61)*B
        else:
            xx = numpy.logspace(-6,0,61)*R
        
        xx2p1 = 1.+xx**2
        kappa = hyp2f1(1.,G/2.,1.5,1./xx2p1)/xx2p1
        c = numpy.isfinite(kappa)
        kappa = kappa[c]
        xx = xx[c]
        model = interpolate.splrep(xx,kappa*xx)
        self.ks = B**2/(2.*interpolate.splint(xx[0],B,model))
        return 2*numpy.pi*self.ks*interpolate.splint(xx[0],R,model)*R0**2


'''class Disk(MassProfile):
    def __init__(self,b=None,rs=None,x=None,y=None,q=None,pa=None):
        self.b = b
        self.rs = rs
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.theta = 0.

    def deflections(self,xin,yin,a=1e-12):
        from numpy import cos,sin,pi
        x,y = self.align_coords(xin,yin,False)

        print self.pa,self.q
        b,rs,q = self.b,self.rs,self.q

        kj = [0.008079,-0.028496,0.045732,-0.027869,0.28762,-0.628801,2.04818,0.563387,-1.415574,0.042244,0.105497]
        sj = [0.110297,0.156972,0.239749,0.383171,0.647549,1.030629,1.599728,2.553228,3.833604,6.305107,9.816998]
        ax = x*0.
        ay = y*0.
        r2 = (q*x)**2+y**2  
        for j in range(len(kj)):
            s = rs*sj[j]
            psi = ((q*s)**2+r2)**0.5
            pref = 2*b*kj[j]*s**2/(psi*((psi+2)**2+(1-q**2)*x**2))
            ax += x*pref*(psi+s*q**2)
            ay += y*pref*(psi+s)
#        ax,ay = self.align_coords(ax,ay,False,True)
        theta = self.pa*pi/180.
        ctheta = cos(theta)
        stheta = sin(theta)
        print ax,ay,ctheta,stheta
        x = ax*ctheta+ay*stheta
        y = ay*ctheta-ax*stheta

        return x,y
        return ax,ay'''


class DPL(MassProfile):
    # cuspy halo model with elliptical potential
    def __init__(self,b=None,rs=None,eta1=None,eta2=None,x=None,y=None,q=None,pa=None):
        self.b = b
        self.rs = rs
        self.eta1 = eta1
        self.eta2 = eta2
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.theta = 0
        self.ks = None

    def deflections(self,xin,yin):
        import numpy
        from scipy import interpolate
        from scipy.special import psi,hyp2f1,beta
        from numpy import cos,sin

        self.gamma=self.eta1 
        self.n=self.eta1+self.eta2 
        if self.n==3:
            #print "warning: singularities at n=3. Please code me"
            self.n+=0.00001     

        q=self.q
        xc,yc = self.align_coords(xin,yin,False)
        r = (q*xc**2+(yc**2)/q)**0.5

        G = self.gamma
        N = self.n

        R0 = self.rs
        B = self.b/R0
  
        drb=2*R0/B*( 
            beta(((N-3.)/2.),((3.-G)/2.)) 
            - 
            beta(((N-3.)/2.),(3./2.)) 
            * 
            (1+B**2)**((3-N)/2.) 
            * 
            hyp2f1( 
                (N-3.)/2., 
                G/2., 
                N/2., 
                1./(1+B**2) 
                ) 
            ) 
        self.ks=self.b/drb
        ks=self.ks

        rr = r/R0

        dr= 2*ks*R0/rr*( 
            beta(((N-3.)/2.),((3.-G)/2.)) 
            - 
            beta(((N-3.)/2.),(3./2.)) 
            * 
            (1+rr**2)**((3-N)/2.) 
            * 
            hyp2f1( 
                (N-3.)/2., 
                G/2., 
                N/2., 
                1./(1+rr**2) 
                ) 
            ) 

        dx,dy = dr*xc*q**0.5/r,dr*yc/r/q**0.5
        ctheta = cos(self.theta)
        stheta = sin(self.theta)
        x = dx*ctheta-dy*stheta
        y = dy*ctheta+dx*stheta

        return x,y
    

class DPL_2(MassProfile):
    # cuspy halo model with elliptical potential
    def __init__(self,b=None,rs=None,eta1=None,x=None,y=None,q=None,pa=None):
        self.b = b
        self.rs = rs
        self.eta1 = eta1
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.theta = 0
        self.ks = None


    def deflections(self,xin,yin):
        import numpy
        from scipy import interpolate
        from scipy.special import psi,hyp2f1,beta
        from numpy import cos,sin

        q=self.q
        xc,yc = self.align_coords(xin,yin,False)
        r = (q*xc**2+(yc**2)/q)**0.5

        #dx = xin-self.x
        #dy = yin-self.y
        #r = (dx**2+dy**2)**0.5

        G = self.eta
        R0 = self.rs
        B = self.b/R0

        xx = numpy.logspace(-6,0,61)*B
        xx2p1 = 1.+xx**2
        # Drop factors of 2 (from Beta term) and 1/2 (missing from Keeton)
        kappa = hyp2f1(1.,G/2.,1.5,1./xx2p1)/xx2p1
        c = numpy.isfinite(kappa)
        kappa = kappa[c]
        xx = xx[c]
        model = interpolate.splrep(xx,kappa*xx)
        ks = B**2/(2*interpolate.splint(xx[0],B,model))
        self.ks = ks

        x = r/R0
        x2 = x**2
        A = numpy.log(1.+x2)/x

        # First x>1
        c = x>=1.
        X = x[c]
        X2 = x2[c]
        if c.sum()>0:
            A[c] += (-self.Gfunc(G/2.,1.5,1./(1.+X2))+psi(1.5)-psi((3-G)/2.))/X

        # Then x<1
        c = x<1.
        X = x[c]
        X2 = x2[c]
        # deal with NFW case (g==1) for x<1
        if G==1 and c.sum()>0:
            G = 0.99
            tmp1 = A[c] -(self.Gfunc(G/2.,(G-1.)/2.,X2/(1.+X2))/X+(X**(2-G))*((1+X2)**((G-3)/2.))*beta((G-3)/2.,1.5)*hyp2f1(1.5,(3-G)/2.,(5-G)/2.,X2/(1+X2)))
            G = 1.01
            tmp2 = A[c] -(self.Gfunc(G/2.,(G-1.)/2.,X2/(1.+X2))/X+(X**(2-G))*((1+X2)**((G-3)/2.))*beta((G-3)/2.,1.5)*hyp2f1(1.5,(3-G)/2.,(5-G)/2.,X2/(1+X2)))
            A[c] = 10**(numpy.log10(tmp1)/2+numpy.log10(tmp2)/2.)
        elif c.sum()>0:
            A[c] += -(self.Gfunc(G/2.,(G-1.)/2.,X2/(1.+X2))/X+(X**(2-G))*((1+X2)**((G-3)/2.))*beta((G-3)/2.,1.5)*hyp2f1(1.5,(3-G)/2.,(5-G)/2.,X2/(1+X2)))

        deflection = ks*R0*A

        dx,dy = deflection*xc*q**0.5/r,deflection*yc/r/q**0.5
        ctheta = cos(self.theta)
        stheta = sin(self.theta)
        x = dx*ctheta-dy*stheta
        y = dy*ctheta+dx*stheta

        return x,y

    def kappa(self,xin,yin):
        import numpy
        from scipy import interpolate
        from scipy.special import psi,hyp2f1,beta
        from numpy import cos,sin

        q=self.q
        xc,yc = self.align_coords(xin,yin,False)
        r = (q*xc**2+(yc**2)/q)**0.5

        G = self.eta
        R0 = self.rs
        B = self.b/R0

        xx = numpy.logspace(-6,0,61)*B
        xx2p1 = 1.+xx**2
        # Drop factors of 2 (from Beta term) and 1/2 (missing from Keeton)
        kappa = hyp2f1(1.,G/2.,1.5,1./xx2p1)/xx2p1
        c = numpy.isfinite(kappa)
        kappa = kappa[c]
        xx = xx[c]
        model = interpolate.splrep(xx,kappa*xx)
        ks = B**2/(2*interpolate.splint(xx[0],B,model))
        self.ks = ks
        xx2p1 = 1.+(r/self.rs)**2.
        kappa = hyp2f1(1.,G/2.,1.5,1./xx2p1)/xx2p1
        kappa*=self.ks

        return kappa

    def Gfunc(self,b,c,z,nmax=100):
        import numpy
        if z.size==0:
            return
        if b==0:
            return z*0.
        top = b
        bot = c
        result = b*z/c
        indx = numpy.arange(z.size)
        for i in range(1,3):
            top *= b+i
            bot *= c+i
            n = i+1.
            term = (top/bot)*(z**n)/n
            result += term
        indx = indx[abs(term/result)>0.0001]
        z = z[indx]
        for i in range(3,nmax):
            top *= b+i
            bot *= c+i
            n = i+1.
            term = (top/bot)*(z**n)/n
            cond = abs(term/result[indx])>0.0001
            result[indx] += term
            if cond.sum()==0:
                break
            z = z[cond]
            indx = indx[cond]
        return result



class eGNFW(MassProfile):
     """
    A subclass for power-law mass models. The `power-law' aspect doesn't
        currently work, but it does work for SIE models.
    """
     def __init__(self,b=None,gammain=None,pa=None,q=None,x=None,y=None,rs=None,trunc=None):
         self.b = b
         self.gammain = gammain
         self.pa = pa
         self.q = q
         self.x = x
         self.y = y
         self.rs = rs
         self.trunc = trunc
         if self.trunc is None:
             self.trunc = 1e-6
         self.gammaout = 3.
         

     def deflections(self,xin,yin):
         from math import pi
         from numpy import arcsin as asin,arcsinh as asinh,logspace,isfinite
         from scipy.special import hyp2f1,beta,gamma
         import cuspFS_simple as cuspFS
         from scipy.interpolate import splrep, splint
         import numpy as np
         ''' normalise kappa -- 
         do this Matt's way as Chae's expression breaks for gamma_in = 1. '''
         
         G = self.gammain
         R0 = self.rs
         B = self.b/R0

         xx = np.logspace(-6,0,61)*B
         xx2p1 = 1.+xx**2
         # Drop factors of 2 (from Beta term) and 1/2 (missing from Keeton)
         kappa = hyp2f1(1.,G/2.,1.5,1./xx2p1)/xx2p1
         c = np.isfinite(kappa)
         kappa = kappa[c]
         xx = xx[c]
         model = splrep(xx,kappa*xx)
         ks = B**2/(2*splint(xx[0],B,model))
         self.ks = ks/2.


         '''gin,gout = self.gammain, self.gammaout
         R=logspace(-5,3,2000)
         r = R/self.rs
         rr2p1 = 1.+r**2.
         kappa = 2. * hyp2f1(gin/2.,(gout-1.)/2.,gout/2.,1./rr2p1)/rr2p1
         c = isfinite(kappa)
         kappa = kappa[c]
         r = r[c]
         model = splrep(r,kappa*2.*pi*r)
         ks = pi * self.b**2. / splint(r[0],self.b,model)
         self.ks = ks'''
         
         ''' ellipticity '''
         e = 1.-self.q
         phi = self.pa*pi/180. - pi/2.
         ''' kappa_0, gammain, gammaout, rs, ellipticity (1-q), phi_0, x, y, trunc '''
         prmt = [self.ks, self.gammain, self.gammaout, self.rs, e, phi,0.,0., self.trunc]
         xout,yout = cuspFS.cuspdeflections(prmt,xin.ravel(),yin.ravel())
         return xout,yout


class eGNFWG(MassProfile):
     """
    Haven't yet got the rotation working right. I.e. Don't use this!
    """
     def __init__(self,b=None,gammain=None,pa=None,q=None,x=None,y=None,rs=None,trunc=None):
         import numpy
         self.b = b
         self.gammain = gammain
         self.pa = pa
         self.q = q
         self.x = x
         self.y = y
         self.rs = rs
         self.gammaout = 3.
         path='/data/ljo31b/cuspEllip'
         self.xmod, self.ymod = numpy.load('%s/cuspModel.dat'%path)
         

     def deflections(self,xin,yin):
         from math import pi,cos,sin
         import numpy
         from numpy import arcsin as asin,arcsinh as asinh,logspace,isfinite
         from scipy.special import hyp2f1,beta,gamma
         import cuspFS_simple as cuspFS
         from scipy.interpolate import splrep, splint
         ''' normalise kappa -- 
         do this Matt's way as Chae's expression breaks for gamma_in = 1. '''
         gin,gout = self.gammain, self.gammaout
         R=logspace(-5,3,2000)
         r = R/self.rs
         rr2p1 = 1.+r**2.
         kappa = 2. * hyp2f1(gin/2.,(gout-1.)/2.,gout/2.,1./rr2p1)/rr2p1
         c = isfinite(kappa)
         kappa = kappa[c]
         r = r[c]
         model = splrep(r,kappa*2.*pi*r)
         ks = pi * self.b**2. / splint(r[0],self.b,model)
         self.ks = ks
         
         x,y = self.align_coords(xin,yin)
         o = numpy.ones(x.size).astype(x.dtype)
         q = self.q
         r0 = self.rs
         g = self.gammain

         eval = numpy.array([abs(x).ravel()/r0,abs(y).ravel()/r0,g*o,q*o]).T
         xout = numpy.sign(x)*(self.ks*self.xmod.eval(eval)*r0).reshape(x.shape)
         yout = numpy.sign(y)*(self.ks*self.ymod.eval(eval)*r0).reshape(y.shape)
         theta = -(self.theta - pi/2.)
         ctheta = cos(theta)
         stheta = sin(theta)
         x = xout*ctheta+yout*stheta
         y = yout*ctheta-xout*stheta
         return x,y

         
