psfFFT = None

def lensModel(inpars,image,sig,gals,lenses,sources,xc,yc,OVRS=1,csub=11,psf=None,noResid=False,verbose=False):
    import pylens,numpy

    model = xc*0.
    for gal in gals:
        gal.setPars(inpars)
        model += gal.pixeval(xc,yc,1./OVRS,csub=csub)

    for src in sources:
        src.setPars(inpars)

    for lens in lenses:
        lens.setPars(inpars)

    model = model + pylens.lens_images(lenses,sources,[xc,yc],1./OVRS)
    if numpy.isnan(model.sum()):
        if verbose==True:
            print 'nan model'
        return -1e300

    if OVRS>1:
        model = iT.resamp(model,OVRS,True)

    if psf is not None:
        from imageSim import convolve
        global psfFFT
        if psfFFT is None:
            psf /= psf.sum()
            model,psfFFT = convolve.convolve(model,psf)
        else:
            model,psfFFT = convolve.convolve(model,psfFFT,False)

    if noResid is True:
        return model
    resid = ((model-image)/sig).ravel()
    if verbose==True:
        print "%f  %5.2f %d %dx%d"%((resid**2).sum(),(resid**2).sum()/resid.size,resid.size,image.shape[1],image.shape[0])
    return -0.5*(resid**2).sum()


def objf(x,lhs,rhs):
    return ((numpy.dot(lhs,x)-rhs)**2).sum()
def objdf(x,lhs,rhs):
    return numpy.dot(lhs.T,numpy.dot(lhs,x)-rhs)

from spasmoid.slsqp import fmin_slsqp
import numpy

def lensFit(inpars,image,sig,gals,lenses,sources,xc,yc,OVRS=1,csub=11,psf=None,mask=None,noResid=False,verbose=False,getModel=False):
    import pylens,numpy

    if psf is not None:
        from imageSim import convolve
        global psfFFT
        if psfFFT is None:
            psf /= psf.sum()
            tmp,psfFFT = convolve.convolve(xc,psf)

    if mask is None or noResid==True or getModel==True:
        mask = image==image

    n = 0
    model = numpy.empty((len(gals)+len(sources),xc[mask].size))
    for gal in gals:
        gal.setPars(inpars)
        gal.amp = 1
        tmp = xc*0.
        tmp[mask] = gal.pixeval(xc[mask],yc[mask],1./OVRS,csub=csub)
        if numpy.isnan(tmp).any():
            if verbose==True:
                print 'nan model'
            return -1e300

        if psf is not None:
            tmp = convolve.convolve(tmp,psfFFT,False)[0]
        if OVRS>1:
            tmp = iT.resamp(model,OVRS,True)
        model[n] = tmp[mask].ravel()
        n += 1

    for lens in lenses:
        lens.setPars(inpars)

    #xc = convolve.convolve(xc,psfFFT,False)[0]
    #yc = convolve.convolve(yc,psfFFT,False)[0]

    x0,y0 = pylens.lens_images(lenses,sources,[xc[mask],yc[mask]],1./OVRS,getPix=True)
    #x0 = convolve.convolve(x0,psfFFT,False)[0]
    #y0 = convolve.convolve(y0,psfFFT,False)[0]
    for src in sources:
        src.setPars(inpars)
        src.amp = 1
        tmp = xc*0.
        tmp[mask] = src.pixeval(x0,y0,1./OVRS,csub=csub)
        if numpy.isnan(tmp).any():
            if verbose==True:
                print 'nan model'
            return -1e300

        if psf is not None:
            tmp = convolve.convolve(tmp,psfFFT,False)[0]
        if OVRS>1:
            tmp = iT.resamp(model,OVRS,True)
        model[n] = tmp[mask].ravel()
        n += 1

    rhs = (image/sig)[mask].flatten()
    op = (model/sig[mask].ravel()).T

    print rhs
    print op
    fit = numpy.array(numpy.linalg.lstsq(op,rhs)[0])
    if (fit<0).any():
        sol = fit
        sol[sol<0] = 0.
        bounds = [(0.,1e31)]*n
        result = fmin_slsqp(objf,sol,bounds=bounds,full_output=1,fprime=objdf,acc=1e-19,iter=2000,args=[op.copy(),rhs.copy()],iprint=0)
        fit,chi = result[:2]
        fit = numpy.asarray(fit)
        fit[fit<0] = 0.

    print fit
    if getModel is True:
        j = 0
        for m in gals+sources:
            m.amp = fit[j]
            j += 1
        return (model.T*fit).T.reshape((n,xc.shape[0],xc.shape[1]))

    if noResid is True:
        model = (model.T*fit).sum(1).reshape(xc.shape)
        j = 0
        for m in gals+sources:
            m.amp = fit[j]
            j += 1
        return model
    model = (model.T*fit).sum(1)
    if mask.all():
        model = model.reshape(xc.shape)
        resid = ((model-image)/sig).ravel()
    else:
        resid = ((model-image[mask])/sig[mask]).ravel()
    if verbose==True:
        print "%f  %5.2f %d %dx%d"%((resid**2).sum(),(resid**2).sum()/resid.size,resid.size,image.shape[1],image.shape[0])
    return -0.5*(resid**2).sum()

