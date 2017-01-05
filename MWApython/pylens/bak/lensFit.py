psfFFT = None


def lensFit(inpars,image,sig,gals,lenses,sources,xc,yc,OVRS=1,csub=5,psf=None,mask=None,noResid=False,verbose=False,getModel=False,showAmps=False,allowNeg=False):
    import pylens,numpy
    import indexTricks as iT
    from imageSim import convolve
    from scipy import optimize

    """
    if psf is not None:
        from imageSim import convolve
        global psfFFT
        if psfFFT is None:
            psf /= psf.sum()
            #if OVRS>1:
            #    from scipy import ndimage
            #    psf = ndimage.zoom(psf,OVRS)
            tmp,psfFFT = convolve.convolve(image,psf)
    """

    if noResid==True or getModel==True:
        mask = None
    if mask is None:
        xin = xc.copy()
        yin = yc.copy()
        imin = image.flatten()
        sigin = sig.flatten()
    else:
        xin = xc[mask].copy()
        yin = yc[mask].copy()
        imin = image[mask].flatten()
        sigin = sig[mask].flatten()

    n = 0
    model = numpy.empty((len(gals)+len(sources),imin.size))
    for gal in gals:
        gal.setPars()
        gal.amp = 1
        if mask is None:
            tmp = gal.pixeval(xin,yin,1./OVRS,csub=csub)
        else:
            tmp = xc*0.
            tmp[mask] = gal.pixeval(xin,yin,1./OVRS,csub=csub)
        if numpy.isnan(tmp).any():
            if verbose==True:
                print 'nan model'
            return -1e300

#        if psf is not None:
#            tmp = convolve.convolve(tmp,psfFFT,False)[0]
        if OVRS>1:
            tmp = iT.resamp(tmp,OVRS,True)
        if psf is not None and gal.convolve is not None:
            tmp = convolve.convolve(tmp,psf,False)[0]
        if mask is None:
            model[n] = tmp.ravel()
        else:
            model[n] = tmp[mask].ravel()
        n += 1

    for lens in lenses:
        lens.setPars()

    x0,y0 = pylens.lens_images(lenses,sources,[xin,yin],1./OVRS,getPix=True)
    for src in sources:
        src.setPars()
        src.amp = 1
        if mask is None:
            tmp = src.pixeval(x0,y0,1./OVRS,csub=csub)
        else:
            tmp = xc*0.
            tmp[mask] = src.pixeval(x0,y0,1./OVRS,csub=csub)
        if numpy.isnan(tmp).any():
            if verbose==True:
                print 'nan model'
            return -1e300

#        if psf is not None:
#            tmp = convolve.convolve(tmp,psfFFT,False)[0]
        if OVRS>1:
            tmp = iT.resamp(tmp,OVRS,True)
        if psf is not None:
            tmp = convolve.convolve(tmp,psf,False)[0]
        if mask is None:
            model[n] = tmp.ravel()
        else:
            model[n] = tmp[mask].ravel()
        n += 1

    rhs = (imin/sigin)
    op = (model/sigin).T

    fit,chi = optimize.nnls(op,rhs)
    """
    fit = numpy.array(numpy.linalg.lstsq(op,rhs)[0])
    if (fit<0).any() and allowNeg==False:
        sol = fit
        sol[sol<0] = 0.
        bounds = [(0.,1e31)]*n
        result = fmin_slsqp(objf,sol,bounds=bounds,full_output=1,fprime=objdf,acc=1e-19,iter=1000*fit.size,args=[op.copy(),rhs.copy()],iprint=0)
        fit,chi = result[:2]
        fit = numpy.asarray(fit)
        fit[fit<0] = 0.
    if showAmps==True:
        print fit
    """

    if getModel is True:
        j = 0
        for m in gals+sources:
            m.amp = fit[j]
            j += 1
        return (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    elif noResid is True:
        model = (model.T*fit).sum(1).reshape(image.shape)
        j = 0
        for m in gals+sources:
            m.amp = fit[j]
            j += 1
        return model
    model = (model.T*fit).sum(1)

    if mask is None:
        model = model.reshape(image.shape)
        resid = ((model-image)/sig).ravel()
    else:
        resid = (model-imin)/sigin

    if verbose==True:
        print "%f  %5.2f %d %dx%d"%((resid**2).sum(),(resid**2).sum()/resid.size,resid.size,image.shape[1],image.shape[0])
    return -0.5*(resid**2).sum()

