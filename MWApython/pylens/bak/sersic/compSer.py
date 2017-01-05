from pylens import MassModels
import numpy,pylab
import indexTricks as iT
from scipy import interpolate
from scipy.special import gamma,gammainc
import ndinterp,time
import sersic
import appSersic
import qsersic

xc = 52.3
yc = 47.8
b = 27.4
qin = 0.999973
#qin = 0.7
b0 = b/qin**0.5
eta = 1.2
reff = 1.
n = 4.

y,x = iT.coords((100,100))

k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
r = ((x-xc)**2+(y-yc)**2)**0.5

amp = b*b*k**(2*n)/(2*n*reff*reff*numpy.exp(k)*(gamma(2*n)*gammainc(2*n,k*(b/reff)**(1/n))))

ae = amp*reff*n*k**(-2*n)*gamma(2*n)
rx = (r/reff)**(1/n)
ar = 2*ae*(1.-(1.-gammainc(2*n,k*rx)))/rx**n
print ae,ar
pylab.plot(r.flatten(),ar.flatten(),'ko')
R = r.flatten()

t = time.time()
for i in range(1):
    dx1,dy1 = sersic.sersicdeflections(x.ravel()-xc,y.ravel()-yc,amp,reff,n,qin)
print time.time()-t
print dx1
t = time.time()
for i in range(10):
    dx1,dy1 = qsersic.sersicdeflections(x.ravel()-xc,y.ravel()-yc,amp,reff,n,qin,m=11)
print time.time()-t
print dx1
pylab.plot(R,(dx1**2+dy1**2).flatten()**0.5,'ro')
pylab.show()
df

t = time.time()
for i in range(1):
    dx2,dy2 = appSersic.sersicdeflections(x.ravel()-xc,y.ravel()-yc,amp,reff,n,qin)
print time.time()-t
print dx2
print dx1/dx2
print dx1/dy2
jj = (dx1**2+dy1**2)**0.5#/(dx2**2+dy2**2)
kk = (dx2**2+dy2**2)**0.5
pylab.plot(r.ravel(),jj,'ro')
pylab.plot(r.ravel(),kk,'bo')
t = time.time()
xp = x-xc
yp = y-yc
q = qin
ax = x*0.
ay = y*0.
#reff = reff/q**0.5
pylab.show()
for i in range(x.shape[0]):
    for j in range(y.shape[1]):
        X = abs(xp[i,j])
        Y = abs(yp[i,j])
        sx = xp[i,j]/X
        sy = yp[i,j]/Y
        ulo = 0.5*(Y/X-X/Y)
        uhi = 0.5*(Y/(X*q*q)-X*q*q/Y)
        off = ulo-1e-3
        u = numpy.logspace(numpy.log10(ulo-off),numpy.log10(uhi-off),171)+off
        u2 = u*u
        rho = (2*u*X*Y+X**2-Y**2)
        rho[0] = 0
        rho = (rho/(1-q**2))**0.5
        kappa = amp*numpy.exp(-k*(rho/reff)**(1./n))
        fx = (1./(1.+u2)**0.5-u/(1.+u2))**0.5
        fy = (1./(1.+u2)**0.5+u/(1.+u2))**0.5
        pylab.plot(u,kappa*fx)
        pylab.show()
        modx = interpolate.splrep(u,kappa*fx)
        ax[i,j] = sx*q*(X*Y)**0.5*interpolate.splint(ulo,uhi,modx)/(1.-q**2)
        mody = interpolate.splrep(u,kappa*fy)
        ay[i,j] = sy*q*(X*Y)**0.5*interpolate.splint(ulo,uhi,mody)/(1.-q**2)


print time.time()-t
print ax
ll = (ax**2+ay**2)**0.5
pylab.plot(r.ravel(),ll.ravel(),'go')
pylab.plot(r.ravel(),ll.ravel()*jj,'mo')
pylab.show()
df



pylab.show()
for i in range(x.shape[0]):
    for j in range(y.shape[1]):
        X = abs(xp[i,j])
        Y = abs(yp[i,j])
        sx = xp[i,j]/X
        sy = yp[i,j]/Y
        ulo = 0.5*(Y/X-X/Y)
        uhi = 0.5*(Y/(X*q*q)-X*q*q/Y)
        u = numpy.linspace(ulo,uhi,301)
        u2 = u*u
        rho = (2*u*X*Y+X**2-Y**2)
        rho[0] = 0
        rho = (rho/(1-q**2))**0.5
        kappa = amp*numpy.exp(-k*(rho/reff)**(1./n))
        fx = (1./(1.+u2)**0.5-u/(1.+u2))**0.5
        fy = (1./(1.+u2)**0.5+u/(1.+u2))**0.5
        modx = interpolate.splrep(u,kappa*fx)
        ax[i,j] = sx*q*(X*Y)**0.5*interpolate.splint(ulo,uhi,modx)/(1.-q**2)
        mody = interpolate.splrep(u,kappa*fy)
        ay[i,j] = sy*q*(X*Y)**0.5*interpolate.splint(ulo,uhi,mody)/(1.-q**2)
print time.time()-t
print ax
ll = (ax**2+ay**2)**0.5
pylab.plot(r.ravel(),ll.ravel(),'go')
pylab.plot(r.ravel(),ll.ravel()*jj,'mo')
pylab.show()
df






pylab.plot(dx)
pylab.show()
print dx.max(),dy.max()
df
dx,dy = d2.deflections(x,y)
print dx
print dy
import sersic
#dx,dy = sersic.sersicdeflections(x,y,b,reff,n,qin)
#df
t = time.time()
for i in range(100):
    dx,dy = d1.deflections(x,y)
print time.time()-t
t = time.time()
for i in range(100):
    dx,dy = d2.deflections(x,y)
print time.time()-t
import plaw

t = time.time()
for i in range(100):
    X = ((x-xc)/b0).flatten()
    Y = ((y-yc)/b0).flatten()

    AX,AY = plaw.plawdeflections(X,Y,b,eta,qin)
    AX = AX.reshape(x.shape)
    AY = AY.reshape(y.shape)
print time.time()-t
df
print AX
print dx
pylab.imshow(AX/dx,origin='lower')
pylab.colorbar()
pylab.show()

df
from scipy import interpolate
u = numpy.logspace(-10,0,501)

x0 = numpy.logspace(-3,1,41)
y0 = numpy.logspace(-3,1,41)
q0 = numpy.linspace(0.1,1.,19)
g0 = numpy.linspace(0.05,1.95,39)

x0 = numpy.logspace(-3,1,21)
y0 = numpy.logspace(-3,1,21)
q0 = numpy.linspace(0.1,1.,10)
g0 = numpy.linspace(0.05,1.95,20)

qg,gg,xg,yg = ndinterp.create_axes_array([q0,g0,x0,y0])

rho = xg**2+yg**2/qg**2
lim = numpy.log10(rho**0.5)
r2 = xg**2+yg**2
dr2 = yg**2-xg**2
xy = (xg*yg)**2
ax = xg*0.
ay = xg*0.
for i in range(q0.size):
    q = q0[i]
    for j in range(g0.size):
        g = g0[j]
        for k in range(x0.size):
            X = x0[k]
            for l in range(y0.size):
                Y = y0[l]
                u = numpy.logspace(lim[i,j,k,l]-5,lim[i,j,k,l],51)
                u2 = u**2
                delta = ((u2*(1-q**2)+dr2[i,j,k,l])**2+4*xy[i,j,k,l])**0.5
                w2 = (delta+r2[i,j,k,l]+u2*(1-q**2))/(delta+r2[i,j,k,l]-u2*(1-q**2))
                kappa = (2-g)*(1/u)**g
                I = kappa*u*w2**0.5/(X**2+(w2*Y)**2)
                mod = interpolate.splrep(u,I)
                ax[i,j,k,l] = X*q*interpolate.splint(u[0],u[-1],mod)
                I *= w2
                mod = interpolate.splrep(u,I)
                ay[i,j,k,l] = Y*q*interpolate.splint(u[0],u[-1],mod)

axes = {}
axes[0] = interpolate.splrep(q0,numpy.arange(q0.size),k=1)
axes[1] = interpolate.splrep(g0,numpy.arange(g0.size),k=1)
axes[2] = interpolate.splrep(x0,numpy.arange(x0.size),k=3)
axes[3] = interpolate.splrep(y0,numpy.arange(y0.size),k=3)

xmod = ndinterp.ndInterp(axes,ax)
ymod = ndinterp.ndInterp(axes,ay)

t = time.time()
for i in range(10):
    X = (x-xc)/b0
    Y = (y-yc)/b0

    X = X.ravel()
    Y = Y.ravel()
    pnts = numpy.array([X*0+qin,X*0+eta,abs(X),abs(Y)]).T
    mx = b0*(xmod.eval(pnts)*numpy.sign(X)).reshape(x.shape)
    my = b0*(ymod.eval(pnts)*numpy.sign(Y)).reshape(x.shape)
print time.time()-t
t = time.time()
for i in range(10):
    dx,dy = d.deflections(x,y)
print time.time()-t
t = time.time()
for i in range(10):
    X = ((x-xc)/b0).flatten()
    Y = ((y-yc)/b0).flatten()

    X2 = X**2
    Y2 = Y**2
    AX = X*0.
    AY = Y*0.
    rho = X2+Y2/qin**2
    lim = numpy.log10(rho**0.5)
    r2 = X2+Y2
    dr2 = Y2-X2
    xy = (X2*Y2)
    q = qin
    for j in range(AX.size):
        u = numpy.logspace(lim[j]-5,lim[j],51)
        u2 = u**2
        delta = ((u2*(1-q**2)+dr2[j])**2+4*xy[j])**0.5
        w2 = (delta+r2[j]+u2*(1-q**2))/(delta+r2[j]-u2*(1-q**2))
        kappa = (2-eta)/u**eta
        I = kappa*u*w2**0.5/(X2[j]+(w2**2*Y2[j]))
        mod = interpolate.splrep(u,I)
        AX[j] = interpolate.splint(u[0],u[-1],mod)
        I *= w2
        mod = interpolate.splrep(u,I)
        AY[j] = interpolate.splint(u[0],u[-1],mod)
    AX = (b0*AX*X*qin).reshape(x.shape)
    AY = (b0*AY*Y*qin).reshape(x.shape)
print time.time()-t
import plaw
t = time.time()
for i in range(10):
    X = ((x-xc)/b0).flatten()
    Y = ((y-yc)/b0).flatten()

    AX,AY = X*0,Y*0
    plaw.plawdeflections(X,Y,b,eta,q,AX,AY)
print AX/dx
print AY/dy
import pylab
pylab.imshow(AX/dx,origin='lower')
#pylab.imshow(numpy.log10(mx/dx),origin='lower')
pylab.colorbar()
pylab.figure()
pylab.imshow(AY/dy,origin='lower')
pylab.colorbar()
pylab.show()

q = numpy.linspace(0.1,1.,19)
n = numpy.linspace(0.4,6.,29)
grid = numpy.empty((40,40,q.size,n.size))


