

def getPSFMatrix(psf,imshape,mask=None):
    """
    Create a PSF matrix given the PSF model and image dimensions
    """
    import numpy
    from scipy.sparse import coo_matrix

    imsize = imshape[0]*imshape[1]
    tmp = numpy.zeros(imshape)
    dx = (imshape[1]-psf.shape[1])/2
    dy = (imshape[0]-psf.shape[0])/2
    tmp[dy:-dy,dx:-dx] = psf.copy()

    cvals = numpy.where(tmp.flatten()!=0)[0]
    pvals = tmp.ravel()[cvals]

    row = numpy.arange(imsize).repeat(cvals.size)
    col = numpy.tile(cvals,imsize)+row
    col -= imsize/2-imshape[0]/2
    pvals = numpy.tile(pvals,imsize)

    good = (col>0)&(col<imsize)
    col = col[good]
    row = row[good]
    pvals = pvals[good]

    pmat = coo_matrix((pvals,(col,row)),shape=(imsize,imsize))
    if mask is not None:
        npnts = mask.sum()
        c = numpy.arange(imsize)[mask.ravel()]
        r = numpy.arange(npnts)
        smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
        pmat = smat.T*(pmat*smat)
    return pmat


def maskPSFMatrix(pmat,mask):
    import numpy
    from scipy.sparse import coo_matrix

    imsize = pmat.shape[0]
    npnts = mask.sum()
    c = numpy.arange(imsize)[mask.ravel()]
    r = numpy.arange(npnts)
    smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
    return smat.T*(pmat*smat)


def getRegularizationMatrix(srcxaxis,srcyaxis,mode="curvature"):
    import numpy
    
    if mode=="zeroth":
        return identity(srcxaxis.size*srcyaxis.size)

    else: from scipy.sparse import diags,csc_matrix,lil_matrix

    if mode=="gradient":
        mat = diags([-2,-2,8,-2,-2],[-srcxaxis.size,-1,0,1,srcxaxis.size],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size))
        mat=lil_matrix(mat)

        #glitches are at left and right edges
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        leftedges=allcols[allcols%srcxaxis.size==0]
        rightedges=allcols[allcols%srcxaxis.size==srcxaxis.size-1]
        for el in leftedges:
            mat[el,el-1]=0
        for el in rightedges:
            if el != allcols.max():
                mat[el,el+1]=0

    elif mode=="curvatureOLD":
        mat=diags([2,2,-8,-8,24,-8,-8,2,2],[-2*srcxaxis.size,-2,-srcxaxis.size,-1,0,1,srcxaxis.size,2*srcxaxis.size,2],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size)) 
        mat=lil_matrix(mat)
        
        #glitches are at left and right edges
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        leftedges=allcols[allcols%srcxaxis.size==0]
        rightedges=allcols[allcols%srcxaxis.size==srcxaxis.size-1]
        leftedgesinone=allcols[allcols%srcxaxis.size==1]
        rightedgesinone=allcols[allcols%srcxaxis.size==srcxaxis.size-2]

        for el in leftedges:
            mat[el,el-1]=0
            mat[el,el-2]=0
        for el in rightedges:
            if el != allcols.max():
                mat[el,el+1]=0
                mat[el,el+2]=0
        for el in leftedgesinone:
            mat[el,el-2]=0
        for el in rightedgesinone:
            if el != allcols.max()-1:
                mat[el,el+2]=0

    elif mode=="curvature":
        I,J=srcxaxis.size,srcyaxis.size
        matrix=lil_matrix((I*J,I*J))
        for i in range(I-2):
            for j in range(J):
                ij=i+j*J
                i1j=ij+1
                i2j=ij+2
                matrix[ij,ij]+=1.
                matrix[i1j,i1j]+=4
                matrix[i2j,i2j]+=1
                matrix[ij,i2j]+=1
                matrix[i2j,ij]+=1
                matrix[ij,i1j]-=2
                matrix[i1j,ij]-=2
                matrix[i1j,i2j]-=2
                matrix[i2j,i1j]-=2
        for i in range(I):
            for j in range(J-2):
                ij=i+j*J
                ij1=ij+J
                ij2=ij+2*J
                matrix[ij,ij]+=1
                matrix[ij1,ij1]+=4
                matrix[ij2,ij2]+=1
                matrix[ij,ij2]+=1
                matrix[ij2,ij]+=1
                matrix[ij,ij1]-=2
                matrix[ij1,ij]-=2
                matrix[ij1,ij2]-=2
                matrix[ij2,ij1]-=2
        for i in range(I):
            iJ_1=i+(J-2)*J
            iJ=i+(J-1)*J
            matrix[iJ_1,iJ_1]+=1
            matrix[iJ,iJ]+=1
            matrix[iJ,iJ_1]-=1
            matrix[iJ_1,iJ]-=1
        for j in range(J):
            I_1j=(I-2)+j*J
            Ij=(I-1)+j*J
            matrix[I_1j,I_1j]+=1
            matrix[Ij,Ij]+=1
            matrix[Ij,I_1j]-=1
            matrix[I_1j,Ij]-=1
        for i in range(I):
            iJ=i+(J-1)*J
            matrix[iJ,iJ]+=1
        for j in range(J):
            Ij=(I-1)+j*J
            matrix[Ij,Ij]+=1
        mat=matrix
    return mat.tocsc()


def getRegularizationMatrix2(srcxaxis,srcyaxis,mode="curvature"):
    N1s=len(srcxaxis)
    N2s=len(srcyaxis)

    mat=diags([2,2,-8,-8,24,-8,-8,2,2],[-2*srcxaxis.size,-2,-srcxaxis.size,-1,0,1,srcxaxis.size,2*srcxaxis.size,2],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size)) 

    if mode=="curvature":
        convolutionmatrix=numpy.array([[0,0,2,0,0],[0,0,-4,0,0],[2,-4,24,-4,2],[0,0,-4,0,0],[0,0,2,0,0]])


def solveForLambda(B,C,Es,Ns,l0,Nfits=1,F=None): 
    from scipy.sparse.linalg import inv,splu
    import numpy
    from scikits.sparse.cholmod import cholesky
    if F is not None:
        print "WARNING: F must be generated using cholesky(Ad) and not cholesky_AAt(Ad); you should check, because the code doesn't"

    for i in range(Nfits):
        L=l0
        A=(B+l0*C).tocsc()
        delta=l0*1e-5
        Ad=(B+(l0+delta)*C).tocsc()
        if F is None:
            T=(2./delta)*((numpy.log(cholesky(Ad).L().diagonal())).sum()-(numpy.log(cholesky(A).L().diagonal())).sum())
        else:
            T=(2./delta)*((numpy.log(cholesky(Ad).L().diagonal())).sum()-(numpy.log(F.L().diagonal())).sum())


        l0=(Ns-l0*T)/(2*Es)
        


        #Old --much slower-- way of doing this.
        #Ainv=inv(A)
        #Tp=(Ainv*C).diagonal().sum()


        #print T/Tp
        
        #Other rearrangments of the non-linear formula
        #l0=Ns/(2*Es+T)
        #l0=(Ns-l0*2*Es)/(T)

    return l0


def getModel(img,var,lmat,pmat,cmat,rmat,reg,niter=10,onlyRes=False):
    from scikits.sparse.cholmod import cholesky
    import numpy

    omat = pmat*lmat
    rhs = omat.T*(img/var)

    B = omat.T*cmat*omat

    res = 0.
    regs = [reg]
    lhs = B+regs[-1]*rmat

    F = cholesky(lhs)
    fit = F(rhs)
    for i in range(niter):
        res = fit.dot(rmat*fit)

        delta = reg*1e3
        lhs2 = B+(reg+delta)*rmat

        T = (2./delta)*(numpy.log(F.cholesky(lhs2).L().diagonal()).sum()-numpy.log(F.L().diagonal()).sum())
        reg = (omat.shape[0]-T*reg)/res
        if abs(reg-regs[-1])/reg<0.005:
            break
        regs.append(reg)
        lhs = B+regs[-1]*rmat
        F = F.cholesky(lhs)
        fit = F(rhs)
    print reg,regs
    res = -0.5*res*regs[-1] + -0.5*((omat*fit-img)**2/var).sum()
    if onlyRes:
        return res,reg
    model = (omat*fit)
    return res,reg,fit,model


def getLensMatrixNN2(lenses,sig,x,y,srcx,srcy):
    """
    Nearest neighbor scheme, not as good as the other function
    """
    import numpy
    from scipy.sparse import coo_matrix
    x = x.copy()
    y = y.copy()

    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        x -= xmap
        y -= ymap

    rvals = numpy.arange(srcx.size)

    rows = []
    cols = []
    for i in range(x.size):
        R2 = ((x[i]-srcx)**2+(y[i]-srcy)**2)
        if R2.min()>0.5:
            continue
        rows.append(rvals[R2.argmin()])
        cols.append(i)
    o = numpy.ones(len(cols))
    return coo_matrix((o,(cols,rows)),shape=(x.size,srcx.size))


def getLensMatrixNN(lenses,sig,x,y,srcx,srcy):
    """
    Nearest neighbor lensing matrix
    """
    import numpy
    from scipy.sparse import coo_matrix
    import special_functions as sf
    x = x.copy()
    y = y.copy()

    # Map (x_src,y_src) to index in S vector
    rvals = numpy.arange(srcx.size)*1.
    infit = numpy.array([srcx,srcy,rvals]).T
    fit = sf.lsqfit(infit,'chebyshev',2,2)

    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        x -= xmap
        y -= ymap

    # Lookup closest index position to mapped (x,y) values
    ind = numpy.round(sf.genfunc(x,y,fit)).astype(numpy.int32)
    c = (ind>=0)&(ind<srcx.size)

    cols = numpy.arange(x.size)[c]
    rows = ind[c]

    return coo_matrix((numpy.ones(rows.size),(cols,rows)),shape=(x.size,srcx.size))


def getLensPoly(lenses,indx,xin,yin,srcx,srcy,sscale=1):
    import numpy
    import pylens
    from scipy.sparse import coo_matrix
    import special_functions as sf
    from math import floor,ceil
    from Polygon import Polygon
    from numpy import round
    import special_functions as sf

    # Map (x_src,y_src) to index in S vector
    rvals = numpy.arange(srcx.size)*1.
    infit = numpy.array([srcx,srcy,rvals]).T
    fit = sf.lsqfit(infit,'chebyshev',2,2)

    npnts = indx.shape[0]

    # Get edges of source grid to find out-of-bounds image pixels
    lxs,hxs = srcx.min(),srcx.max()
    lys,hys = srcy.min(),srcy.max()

    rows = []
    vals = []

    for lens in lenses:
        lens.setPars()
    xl,yl = pylens.getDeflections(lenses,[xin,yin])

    CX = []
    CY = []
    for i in range(npnts):
        # Corner pixels for the i-th center pixel
        x0 = xl[indx[i]]
        y0 = yl[indx[i]]

        # Check if pixel lies (at least in part) out of source plane
        if x0.max()>hxs or x0.min()<lxs or y0.min()<lys or y0.max()>hys:
            continue

        image = Polygon(numpy.array([x0,y0]).T)

        # Find subgrid of source plane pixels
        xlo = srcx[srcx<x0.min()].max()
        xhi = srcx[srcx>x0.max()].min()
        ylo = srcy[srcy<y0.min()].max()
        yhi = srcy[srcy>y0.max()].min()

        # loop over all pixels in subgrid to calculate overlap area
        val = []
        for x in numpy.arange(xlo,xhi+sscale/2,sscale):
            for y in numpy.arange(ylo,yhi+sscale/2,sscale):
                src = Polygon(numpy.array([[x,x+sscale,x+sscale,x],[y,y,y+sscale,y+sscale]]).T)

                cross = image&src
                c = cross.area()
                if c>0:
                    CX.append(x)
                    CY.append(y)
                    rows.append(i)
                    val.append(c)
        val = numpy.array(val)
        vals = vals+(val/val.sum()).tolist()
    cols = numpy.round(sf.genfunc(CX,CY,fit)).astype(numpy.int32)

    """
    import pylab as plt
    plt.scatter(rows,cols)
    plt.show(block=True)
    print len(rows)
    print npnts*srcx.size
    """

    return coo_matrix((vals,(rows,cols)),shape=(npnts,srcx.size))


def PixelNumber(x0,y0,xsrcaxes,ysrcaxes,mode='NearestCentre'):
    import numpy
    srcpixscale=xsrcaxes[1]-xsrcaxes[0]
    if mode=='NearestCentre':
        xpixelnumber=(numpy.floor(((x0-xsrcaxes[0])/srcpixscale)+0.5))
        ypixelnumber=(numpy.floor(((y0-ysrcaxes[0])/srcpixscale)+0.5))

    if mode=='NearestBottomLeft':
        xpixelnumber=(numpy.floor(((x0-xsrcaxes[0])/srcpixscale)))
        ypixelnumber=(numpy.floor(((y0-ysrcaxes[0])/srcpixscale)))

    pixelnumber=ypixelnumber*len(xsrcaxes)+xpixelnumber

    pixelnumber[xpixelnumber<0]=-1
    pixelnumber[ypixelnumber<0]=-1
    pixelnumber[xpixelnumber>=len(xsrcaxes)]=-1
    pixelnumber[ypixelnumber>=len(ysrcaxes)]=-1

    if mode=='NearestBottomLeft': 
        #we want the point to be within the grid
        pixelnumber[xpixelnumber==len(xsrcaxes)-1]=-1
        pixelnumber[ypixelnumber==len(ysrcaxes)-1]=-1

    return pixelnumber

def debugPixelNumber(srcxaxis,srcyaxis,srcx,srcy):
    """
    Function to show that PixelNumber is doing the correct thing
    """
    import numpy
    import pylab as plt
    for i in range(10):
     for j in range(10):
        a=45.36+(numpy.random.rand()-0.5)*3
        b=49.5+(numpy.random.rand()-0.5)*3

        srcpixscale=srcxaxis[1]-srcxaxis[0]
        a=srcxaxis[0]+0.05+i*srcpixscale
        b=srcyaxis[0]+0.05+j*srcpixscale

        #R= list(PixelNumber([a],[b],srcxaxis,srcyaxis,mode='NearestTopLeft'))
        R= list(PixelNumber([a],[b],srcxaxis,srcyaxis,mode='NearestCentre'))
        
        R=R[0]
        
        plt.scatter(srcx,srcy,s=1)
        
        plt.scatter([a],[b],c='r')


        plt.scatter(srcx[R],srcy[R],c='orange')
        plt.scatter(srcx[R+srcyaxis.size],srcy[R+srcyaxis.size])
        plt.scatter( srcx[R+srcyaxis.size+1],srcy[R+srcyaxis.size+1])

        plt.show(block=True)
        
     return None


def getLensMatrixNNTom(lenses,x,y,srcxaxis,srcyaxis,srcx,srcy):
    """
    Nearest neighbor lensing matrix, using floor
    """
    import numpy
    from scipy.sparse import coo_matrix
    xin = x.copy()
    yin = y.copy()

    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        xin -= xmap
        yin -= ymap

    PN=PixelNumber(xin,yin,srcxaxis,srcyaxis)
    row=numpy.arange(len(PN))
    row=row[PN>-1]
    PN=PN[PN>-1]
    o=numpy.ones(len(PN))
    """
    import pylab as plt
    plt.scatter(row,PN)
    plt.show(block=True)
    print len(row)
    """
    return coo_matrix((o,(row,PN)),shape=(x.size,srcxaxis.size*srcyaxis.size))

def BilinearWeight(xi,yi,xs,ys,srcpixscale):
    import numpy
    scaledxdif = (numpy.abs(xi-xs)/srcpixscale)
    scaledydif = (numpy.abs(yi-ys)/srcpixscale)
    xweight=1-scaledxdif
    yweight=1-scaledydif
    xweight[xweight<0.]=0
    yweight[yweight<0.]=0
    return xweight*yweight


def getLensMatrixBilinearTEC(lenses,x,y,srcx,srcy,srcxaxis,srcyaxis,imsize):
    """
    Bilinear lensing matrix
    """
    import numpy
    from scipy.sparse import coo_matrix

    xin = x.copy()
    yin = y.copy()

    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        xin -= xmap
        yin -= ymap

    BL=PixelNumber(xin,yin,srcxaxis,srcyaxis,mode='NearestBottomLeft')
    row=numpy.arange(len(BL))[BL>-1]
    xin,yin=xin[BL>-1],yin[BL>-1]
    BL=BL[BL>-1]
    BR,TL,TR=BL+1,BL+srcxaxis.size,BL+srcxaxis.size+1
    row=list(numpy.concatenate((row,row,row,row)))
    PN=list(numpy.concatenate((BL,BR,TL,TR)))
    xin=numpy.concatenate((xin,xin,xin,xin))
    yin=numpy.concatenate((yin,yin,yin,yin))

    #now calculate bilinear coefficients
    srcpixscale=srcxaxis[1]-srcxaxis[0]

    Pweight=list(BilinearWeight(xin,yin,srcx[PN],srcy[PN],srcpixscale))

    return coo_matrix((Pweight,(row,PN)),shape=(x.size,srcxaxis.size*srcyaxis.size))


def getLensMatrixBilinear(lenses,x,y,srcx,srcy,srcxaxis,srcyaxis,imsize):
    """
    Bilinear lensing matrix
    """
    import numpy
    from scipy.sparse import coo_matrix

    xin = x.copy()
    yin = y.copy()

    # pylens.getDeflections() would do this too....
    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        xin -= xmap
        yin -= ymap

    scale = srcxaxis[1]-srcxaxis[0]
    spix = PixelNumber(xin,yin,srcxaxis,srcyaxis,mode='NearestBottomLeft')
    c = spix>-1
    row = numpy.arange(spix.size)[c]
    xin,yin = xin[c],yin[c]
    spix = spix[c].astype(numpy.int32)

    # Create row, col, value arrays
    r = numpy.empty(row.size*4)
    c = numpy.empty(row.size*4)
    w = numpy.empty(row.size*4)

    # These are the lower-right pixel weights
    r[:row.size] = row
    c[:row.size] = spix
    w[:row.size] = (1.-numpy.abs(xin-srcx[spix])/scale)*(1.-numpy.abs(yin-srcy[spix])/scale)

    # Now do the lower-left, upper-left, and upper-right pixels
    a = [1,srcxaxis.size,-1]
    for i in range(1,4):
        spix += a[i-1]
        r[i*row.size:(i+1)*row.size] = row
        c[i*row.size:(i+1)*row.size] = spix
        w[i*row.size:(i+1)*row.size] = (1.-numpy.abs(xin-srcx[spix])/scale)*(1.-numpy.abs(yin-srcy[spix])/scale)

    return coo_matrix((w,(r,c)),shape=(x.size,srcx.size))


def getLensMatrixRayshooting(lenses,x,y,srcx,srcy,srcxaxis,srcyaxis):
    """
    Multi-ray shooting lensing matrix. Does not work
    """

    print "I don't work (or even really exist)"
    exit()
    import numpy
    import ndinterp
    from scipy.sparse import coo_matrix

    x0,y0 = x.copy(),y.copy()

    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        x0 -= xmap
        y0 -= ymap
      

    pixelnumber=PixelNumber(xin,yin,srcxaxis,srcyaxis)
    row=numpy.arange(len(pixelnumber))
    row=row[pixelnumber>-1]
    pixelnumber=pixelnumber[pixelnumber>-1]
    ones=numpy.ones(len(pixelnumber))

    return coo_matrix((ones,(row,pixelnumber)),shape=(x.size,srcxaxis.size*srcyaxis.size))

def getLensPolyTom(lenses,indx,xin,yin,srcx,srcy,sscale=1):
    import numpy
    from scipy.sparse import coo_matrix
    assert 1==2
    x0,y0 = x.copy(),y.copy()

    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        x0 -= xmap
        y0 -= ymap


