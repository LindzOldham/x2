import glob,numpy,pylab,pyfits,cPickle

def clip(arr,nsig=3.5):
    a = arr.flatten()
    a.sort()
    a = a[a.size*0.05:a.size*0.8]
    m,s,l = a.mean(),a.std(),a.size
    while 1:
        a = a[abs(a-m)<s*nsig]
        if a.size==l:
            return m,s
        m,s,l = a.mean(),a.std(),a.size


def colorImage(b,g,r,bMinusr=0.8,bMinusg=0.4,sdev=None,nonlin=5.,m=0.5,M=None):
    w = r.shape[0]/2-5
    rb = r/b
    gb = g/b
    rnorm = numpy.median(rb[w:-w,w:-w])
    gnorm = numpy.median(gb[w:-w,w:-w])
    r /= rnorm
    g /= gnorm
    r *= 10**(0.4*bMinusr)
    g *= 10**(0.4*bMinusg)

    r /= 620.
    g /= 540.
    b /= 460.

    I = (r+g+b)/3.

    if sdev is None:    
        sdev = clip(I)[1]
    m = m*sdev
    if M is None:
        M = I[w:-w,w:-w].max()
    nonlin = nonlin*sdev

    f = numpy.arcsinh((I-m)/nonlin)/numpy.arcsinh((M-m)/nonlin)
    f[I<m] = 0.
    f[I>M] = 1.
    R = r*f/I
    G = g*f/I
    B = b*f/I

    R[I<=0] = 0.
    G[I<=0] = 0.
    B[I<=0] = 0.

    R[R<=0] = 0.
    G[G<=0] = 0.
    B[B<=0] = 0.

    R[R>1] = 1.
    G[G>1] = 1.
    B[B>1] = 1.

    white = True
    if white:
        cond = (f==1)
        R[cond] = 1.
        G[cond] = 1.
        B[cond] = 1.

    arr = numpy.empty((R.shape[0],R.shape[1],3))
    arr[:,:,0] = R
    arr[:,:,1] = G
    arr[:,:,2] = B

    return arr,sdev,M,rnorm,gnorm
