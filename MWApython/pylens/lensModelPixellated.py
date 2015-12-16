psfFFT = None

def setupImageCoords(image,sig,psf,mask=None,regularizationmode='curvature'):
    import indexTricks as iT
    from imageSim import convolve
    from scipy.sparse import coo_matrix,diags
    import pixellatedTools
    import numpy
    inshape = image.shape
    image = image.flatten()
    sig = sig.flatten()
    ysize,xsize = inshape

    # There are two sets of coordinates in the image plane, one for the
    #  centers of pixels and one for the corners. This defines a mapping that
    #  describes which corners are associated with a center.
    yc,xc = iT.coords((ysize+1,xsize+1)) # Corners
    yc = yc.flatten()-0.5
    xc = xc.flatten()-0.5
    y0,x0 = iT.coords((ysize,xsize))     # Centers
    y0 = y0.flatten()
    x0 = x0.flatten()
    
    i = numpy.arange(x0.size)
    res = i%xsize
    p = i/xsize
    i = p*(xsize+1)+res
    indx = numpy.array([i,i+1,i+xsize+1,i+xsize+2]).T


    P = pixellatedTools.getPSFMatrix(psf,inshape)
    

    if mask!=None:
        image=image[mask]
        indx=None
        sig=sig[mask]
        yc=yc[mask]
        xc=xc[mask]
        y0=y0[mask]
        x0=x0[mask]
        P= P.tocsc()[:,list(mask)[0]]
        P= P[list(mask)[0],:]

    C = diags(1./sig**2,0)
    c= diags(sig**2,0)

    return image,sig,yc,xc,y0,x0,P,c,C,indx,inshape,mask

def setupSourceGrid(sshape,pscale,scentre,mask=None,regularizationmode='curvature',R=None):
    import pixellatedTools
    import indexTricks as iT

    srcy,srcx = iT.coords(sshape)*pscale
    srcx -= srcx.mean()
    srcy -= srcy.mean()
    srcy += scentre[1]
    srcx += scentre[0]

    srcxaxis=srcx[0]
    srcyaxis=srcy[:,0]

    srcx = srcx.flatten()

    srcy = srcy.flatten()

    if R==None:
        R= pixellatedTools.getRegularizationMatrix(srcxaxis,srcyaxis,mode=regularizationmode)

    return srcx,srcy,srcxaxis,srcyaxis,R,pscale,sshape

def lensModel(imagesetup,sourcesetup,lenses,gals,regconst,csub=11,psf=None,noResid=False,mapping='polygon',outputimages=False,returnRegCoeffs=False,mask2=None):
    image,sig,yc,xc,y0,x0,P,c,C,indx,imshape,mask=imagesetup
    srcx,srcy,srcxaxis,srcyaxis,R,pscale,sshape=sourcesetup
    import pixellatedTools
    import numpy
    import indexTricks as iT
    from imageSim import convolve
    from scipy.sparse.linalg import lsmr
    from scipy.sparse import diags

    galsubimage = image*1.
    for gal in gals:
        gal.setPars()      
        galsubimage -= P*gal.pixeval(x0,y0,csub=csub)

    galsum=(image-galsubimage)#.reshape(imshape)

    for lens in lenses:
        lens.setPars()

    if mapping =='polygon':
        Lm = pixellatedTools.getLensPoly(lenses,indx,xc,yc,srcx,srcy,pscale)
    if mapping =='bilinear':
        Lm = pixellatedTools.getLensMatrixBilinear(lenses,x0,y0,srcx,srcy,srcxaxis,srcyaxis)
    if mapping =='nearest':
        Lm = pixellatedTools.getLensMatrixNNTom(lenses,x0,y0,srcx,srcy,srcxaxis,srcyaxis)

    if regconst!=0:
        Sinv=diags([regconst],[0],shape=R.shape)*R

    Om = P*Lm
    B=Om.T*C*Om

    rhs = Om.T*(galsubimage/sig**2)
    if regconst!=0:lhs = (B)+Sinv
    else: lhs=B

    #old slow method, not using cholesky decomposition
    #fit = lsmr(lhs,rhs,maxiter=5000)
    #fit=fit[0]

    from scikits.sparse.cholmod import cholesky_AAt
    A=lhs
    F=cholesky_AAt((A.T).tocsc())
    fit = F(A.T * rhs)

    imin=image.copy()
    s=sig.copy()
    im=numpy.zeros(imshape).flatten()

    if mask != None:
        s=numpy.ones(imshape).flatten()*1e-99
        s[mask]=sig
        imin=numpy.zeros(imshape).flatten()
        imin[mask]=image

    s=s.reshape(imshape)
    imin=imin.reshape(imshape)
    
    if outputimages:
        import pyfits
        im[mask]+= (Om*fit)
        im[mask]+=galsum
        im=im.reshape(imshape)
        src=fit.reshape(sshape).copy()
        pyfits.PrimaryHDU(im).writeto("model.fits",clobber=True)
        pyfits.PrimaryHDU((imin-im)/s).writeto("resid.fits",clobber=True)
        pyfits.PrimaryHDU(src).writeto("src.fits",clobber=True)
        sig=sig.flatten()


    #if noResid: return [(Om*fit).reshape(imshape),fit]
    
    resid=((Om*fit)-galsubimage)/sig


    #this prunes out the inner region of the fit - i.e. the regime where the regularization has no impact.
    if mask2!=None:
        galsubimage=galsubimage[mask2]
        Om=Om.tocsc()[mask2[0],:]
        C=diags(1./(sig[mask2])**2,0)


    s=numpy.matrix(fit).T
    #d=numpy.matrix(galsubimage).T
    #f=Om
    #Cdinv=C

    Ed=0.5*(resid**2).sum()
    Es=(0.5*((s.T)*(R*s)))[0,0]


    M=-(Ed+regconst*Es)

    return M,Es,B,R,F

def regularizeLensModelOLD(imagesetup,sourcesetup,lenses,gals,csub=11,psf=None,noResid=False,mapping='polygon',outputimages=False,returnRegCoeffs=False,mask2=None,accuracy=0.01,regconst=0):
    import pixellatedTools
    import numpy
    from scipy.optimize import curve_fit


    sshape=sourcesetup[-1]
    Ns=sshape[0]*sshape[1]

    Nt=5
    for i in range(20):
        regconstlist=[regconst]
        i=0
        while i<Nt:
            i+=1
            res,Es,B,R,F=lensModel(imagesetup,sourcesetup,lenses,gals,regconst,csub=11,psf=None,noResid=False,mapping=mapping,outputimages=False,returnRegCoeffs=False,mask2=None)
            regconst=pixellatedTools.solveForLambda(B,R,Es,Ns,regconst,Nfits=1)
            regconstlist.append(regconst)


            #print regconst
            
            try:
                if numpy.abs(regconstlist[-1]-regconstlist[-2])>numpy.abs(regconstlist[-2]-regconstlist[-3]):
                    regconst=regconstlist[-1]+(regconstlist[-1]-regconstlist[-2])*10
                    regconstlist=[regconst]
                    #print "jump",regconst
                    i=0
                    continue

                if numpy.abs((regconstlist[-3]-regconstlist[-1])/regconstlist[-1])<0.005:
                    if Nt>10:
                        import pylab as plt
                        plt.plot(regconstlist)
                        plt.show(block=True)

                    return regconst,res
            except IndexError: continue

        x=numpy.arange(Nt+1)
        y=numpy.array(regconstlist)

        """
        def func(x, a, b):
            return a*numpy.arctan(x-b)

        popt, pcov = curve_fit(func, x, y)
        a,b=popt

        regconst=func(1000000,a,b) 
        print "t",regconst
        """

        def straightline(x, a, b):
            return a*x+b

        def quadratic(x, a, b, c):
            return a*x**2+b*x+c

        def getEnd(x1,x2,x3,x4,x5,x6):
            dx1 = x2-x1
            dx2 = x3-x2
            dx3 = x4-x3
            dx4 = x5-x4
            dx5 = x6-x5


            reldifs=1./numpy.array([dx2/dx3,dx3/dx2,dx4/dx3,dx5/dx4])
            import pylab as plt


            x=numpy.arange(len(reldifs))
            popt, pcov = curve_fit(straightline, x, reldifs)
            a,b=popt
            #print a,b

            #plt.plot(reldifs)
            #plt.plot(x,straightline(x,a,b))
            #plt.show(block=True)
            


            dxI=dx5
            dxlist=[]
            for I in range(100):
                i=I+3
                dxN=dxI/straightline(i,a,b)
                if dxN<0:break
                dxlist.append(dxN)
                dxI=dxN

            return x6+sum(dxlist)

        """
        def getEnd(x1,x2,x3,x4):
            dx1 = x2-x1
            dx2 = x3-x2
            dx3 = x4-x2

            if dx3>dx2:
                print "jump"
                return x4+3*dx3

            offs = [dx1,dx2,dx3]

            #if dx2-dx1 >0 and dx3>0:
            #    alpha=numpy.log(dx3)/numpy.log(dx2-dx1)
            #elif dx2-dx1 <0 and dx3<0:
            #    alpha=-numpy.log(-dx3)/numpy.log(-dx2+dx1)
            #else: 
            #    print i,x1,x2,x3,x4
            #    return x4

            if dx2>0 and dx1>0:
                alpha=numpy.log(dx2)/numpy.log(dx1)
            elif dx2<0 and dx1<0:
                alpha=-numpy.log(-dx2)/numpy.log(-dx1)
            else: 
                print i,x1,x2,x3,x4
                return x4


            while dx3>dx2:
                #offs.append((dx2-dx1)**1.115)
                try:
                    offs.append((dx3-dx2)**alpha)
                except FloatingPointError:
                    return x1+sum(offs)
                dx2 = dx3
                dx3 = offs[-1]
                


            return x1+sum(offs)
            """

        regconst=getEnd(regconstlist[0],regconstlist[1],regconstlist[2],regconstlist[3],regconstlist[4],regconstlist[5])
        #"""



    return regconst,res


def regularizeLensModel(imagesetup,sourcesetup,lenses,gals,csub=11,psf=None,noResid=False,mapping='polygon',outputimages=False,returnRegCoeffs=False,mask2=None,accuracy=0.01,regconst=0):
    import pixellatedTools
    import numpy
    from scipy.optimize import curve_fit
    import time
    t0=time.clock()
    sshape=sourcesetup[-1]
    Ns=sshape[0]*sshape[1]

    printflag=False


    Nt=2
    for i in range(20):
        regconstlist=[regconst]
        i=0
        while i<Nt:
            i+=1
            res,Es,B,R,F=lensModel(imagesetup,sourcesetup,lenses,gals,regconst,csub=11,psf=None,noResid=False,mapping=mapping,outputimages=False,returnRegCoeffs=False,mask2=None)
            regconst=pixellatedTools.solveForLambda(B,R,Es,Ns,regconst,Nfits=1)
            regconstlist.append(regconst)

            


            #print regconst

        x0=regconstlist[-3]
        x1=regconstlist[-2]
        x2=regconstlist[-1]

        denominator = x2 - 2*x1 + x0

        aitkenX = x2 - ( (x2 - x1)**2 )/ denominator


        regconst=aitkenX


        if printflag==True:print x0,x1,x2,"ax:", aitkenX

        if regconst<0:
            #print x0,x1,x2,"ax:", aitkenX
            #print x2/2.
            regconst=x2/2.
            """#This is a robustness checker - the aitken code passed everything so far.
            regconst2=x2
            j=0
            while j<20:
                j+=1
                res,Es,B,R,F=lensModel(imagesetup,sourcesetup,lenses,gals,regconst2,csub=11,psf=None,noResid=False,mapping=mapping,outputimages=False,returnRegCoeffs=False,mask2=None)
                regconst2=pixellatedTools.solveForLambda(B,R,Es,Ns,regconst2,Nfits=1)
                print regconst2
            printflag=True
            print regconst
            """

        elif numpy.abs(aitkenX - x2)/aitkenX<0.005:
            return regconst,res
            

    print "I haven't had any luck with this one"
    return None
