import numpy


def drawRandom(pix,nsamps):
    from scipy import interpolate

    apix = pix.argsort()
    spix = numpy.sort(pix)

    cpix = spix.cumsum()
    cpix -= cpix[0]
    cpix /= cpix[-1]

    model = interpolate.splrep(cpix,numpy.arange(cpix.size),s=1)
    dpix = interpolate.splev(numpy.random.random(nsamps),model).astype(numpy.int32)

    A = numpy.unique(dpix)
    while A.size<nsamps:
        dpix = interpolate.splev(numpy.random.random(nsamps),model).astype(numpy.int32)
        A = numpy.unique(numpy.concatenate((dpix,A)))
    if A.size>nsamps:
        A = A[numpy.random.random(A.size).argsort()[:nsamps]]
    return apix[A]


def drawRandom2(pix,nsamps):
    apix = pix.argsort()
    spix = numpy.sort(pix)

    spix /= spix.min()
    spix = (1.+spix)/2.
    r = numpy.random.random(spix.size)*spix
    return apix[r.argsort()[-nsamps:]]


def drawRandom3(pix,nsamps):
    from scipy import interpolate
    apix = pix.argsort()
    spix = numpy.sort(pix)

    spix /= spix.sum()
    spix = spix**0.5

    cpix = spix.cumsum()
    cpix -= cpix[0]
    cpix /= cpix[-1]

    model = interpolate.splrep(cpix,numpy.arange(cpix.size),s=1)
    dpix = interpolate.splev(numpy.random.random(nsamps),model).astype(numpy.int32)

    A = numpy.unique(dpix)
    while A.size<nsamps:
        dpix = interpolate.splev(numpy.random.random(nsamps),model).astype(numpy.int32)
        A = numpy.unique(numpy.concatenate((dpix,A)))
    if A.size>nsamps:
        A = A[numpy.random.random(A.size).argsort()[:nsamps]]
    return apix[A]


def drawRandom4(pix,nsamps):
    from scipy import interpolate
    apix = pix.argsort()
    spix = numpy.sort(pix)

    spix /= spix.sum()
    spix = spix**0.5

    spix /= spix.min()
    spix = (1.+spix)/2.
    r = numpy.random.random(spix.size)*spix
    return apix[r.argsort()[-nsamps:]]


import time,pylab
s1 = numpy.empty(0)
s2 = numpy.empty(0)
s3 = numpy.empty(0)
s4 = numpy.empty(0)
import indexTricks as iT

y,x = iT.coords((11,11))
y -= y.mean()
x -= x.mean()

p = 10*numpy.exp(-0.5*(x**2+y**2)/9.)+x+y-x.min()-y.min()
p = p.flatten()
p.sort()

t = time.time()
for i in range(1500):
    s1 = numpy.concatenate((s1,drawRandom(p,20)))
print time.time()-t

t = time.time()
for i in range(1500):
    s2 = numpy.concatenate((s2,drawRandom2(p,20)))
print time.time()-t

t = time.time()
for i in range(1500):
    s3 = numpy.concatenate((s3,drawRandom3(p,20)))
print time.time()-t

t = time.time()
for i in range(1500):
    s4 = numpy.concatenate((s4,drawRandom4(p,20)))
print time.time()-t




import pylab
pylab.hist(s1,s1.max()-s1.min(),alpha=0.5)
pylab.hist(s2,s2.max()-s2.min(),alpha=0.4)
pylab.hist(s3,s3.max()-s3.min(),alpha=0.3)
pylab.hist(s4,s4.max()-s4.min(),alpha=0.25)
pylab.plot(p*s2.size/p.sum())
k = p**0.5
pylab.plot(k*s2.size/k.sum())
#pylab.figure()
#pylab.imshow(p.reshape((11,11)),origin='lower',interpolation='nearest')
#pylab.colorbar()
pylab.show()
