import numpy

def getPSFMatrix(psf,imshape,mask=None):
    """
    Create a PSF matrix given the PSF model and image dimensions
    """
    import numpy
    from scipy.sparse import coo_matrix
    import indexTricks as iT


    imsize = imshape[0]*imshape[1]

    y,x = iT.coords(psf.shape)
    x -= int(x.mean())
    y -= int(y.mean())
    c = psf.ravel()!=0
    x = x.ravel().astype(numpy.int32)[c]
    y = y.ravel().astype(numpy.int32)[c]
    Y,X = iT.coords(imshape)
    X = X.ravel().astype(numpy.int32)
    Y = Y.ravel().astype(numpy.int32)

    cols = X.repeat(x.size)+numpy.tile(x,X.size)
    rows = Y.repeat(y.size)+numpy.tile(y,Y.size)

    C = (cols>=0)&(cols<imshape[1])&(rows>=0)&(rows<imshape[0])
    cols = cols[C]
    rows = rows[C]

    pvals = numpy.tile(psf.ravel()[c],imsize)[C]
    col = cols+rows*imshape[1]
    row = numpy.arange(imsize).repeat(c.sum())[C]

    pmat = coo_matrix((pvals,(col,row)),shape=(imsize,imsize))
    if mask is not None:
        a = numpy.arange(mask.size)
        cols = a[mask.ravel()]
        pmat = pmat.tocsr()[:,cols].tocsc()[cols]
#        npnts = mask.sum()
#        c = numpy.arange(imsize)[mask.ravel()]
#        r = numpy.arange(npnts)
#        smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
#        pmat = smat.T*(pmat*smat)
    return pmat


def maskPSFMatrix(pmat,mask):
    import numpy
    from scipy.sparse import coo_matrix,diags,lil_matrix

    a = numpy.arange(mask.size)
    cols = a[mask.ravel()]
    J = pmat.tocsr()[:,cols].tocsc()[cols]
    return J

"""
def bob():
    c = diags(mask.ravel()*1.,0)
    J = c*(maspmat*c)
    return lil_matrix(J

    imsize = pmat.shape[0]
    npnts = mask.sum()
    c = numpy.arange(imsize)[mask.ravel()]
    r = numpy.arange(npnts)
    smat = coo_matrix((numpy.ones(npnts)*1.,(c,r)),shape=(imsize,npnts))
    return smat.T*(pmat*smat)
"""

def getRegularizationMatrix(srcxaxis,srcyaxis,mode="curvature"):
    import numpy
    if type(srcxaxis)==type(1):
        srcxaxis = numpy.arange(srcxaxis)
        srcyaxis = numpy.arange(srcyaxis)

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

import time
def fastMult(a,b):
    import numpy
    import matmult

    a = a.tocsr()
    b = b.tocsr()

    M,K1 = a.shape
    K1,N = b.shape

    iw = numpy.empty(max(M,K1,N),dtype=numpy.intc)

    t = time.time()
    nnz = matmult.count(M,N,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    print time.time()-t
    t = time.time()
    c,jc,ic = matmult.multd(M,N,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    print time.time()-t
    return a.__class__((c,jc,ic),shape=(M,N))


def fastAAT1(a):
    import numpy
    import matmult

    b = a.tocsr()
    a = a.T.tocsr()

    M,N = a.shape
    t = time.time()
    b.sort_indices()
    print time.time()-t

    iw = numpy.empty(max(a.shape),dtype=numpy.intc)
    nnz = matmult.count_t(M,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    c,jc,ic = matmult.multd_t(M,M,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    out = a.__class__((c,jc,ic),shape=(M,M))
    return out+out.T


def fastAAT2(a):
    import numpy
    import matmult

    b = a.tocsr()
    a = a.T.tocsr()

    M,N = a.shape
    t = time.time()
    matmult.sort_csr(b.indptr.size,b.indptr,b.indices,b.data)
    print time.time()-t

    iw = numpy.empty(max(a.shape),dtype=numpy.intc)
    nnz = matmult.count_t(M,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    c,jc,ic = matmult.multd_t(M,M,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    out = a.__class__((c,jc,ic),shape=(M,M))
    return out+out.T


def fastAAT3(a):
    import numpy
    import matmult
    import radSort

    b = a.tocsr()
    a = a.T.tocsr()

    M,N = a.shape
#    t = time.time()
#    radSort.sort_csr(b.indptr,b.indices,b.data,M)
#    print time.time()-t
    t = time.time()
#    wd = numpy.zeros(max(M,N))
#    wi = wd.astype(numpy.int32)
#    matmult.sort_csr3(b.indptr.size,b.indptr,b.indices,b.data,wi,wd)
    matmult.sort_csr3(b.indptr.size,N,b.indptr,b.indices,b.data)
    print time.time()-t

    iw = numpy.empty(max(a.shape),dtype=numpy.intc)
    nnz = matmult.count_t(M,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    c,jc,ic = matmult.multd_t(M,M,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    out = a.__class__((c,jc,ic),shape=(M,M))
    return out+out.T


def bah():
    b1 = b.copy()
    b2 = b.copy()
    b3 = b.copy()

    t = time.time()
    b1.sort_indices()
    print time.time()-t,'python'
    print b1.indices

    t = time.time()
    matmult.sort_csr(b2.indptr.size,b2.indptr,b2.indices,b2.data)
    print time.time()-t,'subsort'
    print b2.indices

    import radSort
    print numpy.sort(b3.indices),b3.indices.size,b.shape
    t = time.time()
    #matmult.sort_csr2(b3.indptr.size,b3.data.size,b3.indptr,b3.indices,b3.data)
    radSort.sort_csr(b3.indptr,b3.indices,b3.data,M)
    print time.time()-t,'qsort'
    print b3.indices
    print abs((b3.data-b2.data)/b2.data).max()
    df

    iw = numpy.empty(max(a.shape),dtype=numpy.intc)
    t = time.time()
    nnz = matmult.count_t(M,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    print time.time()-t
    t = time.time()
    c,jc,ic = matmult.multd_t(M,M,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    print time.time()-t
    out = a.__class__((c,jc,ic),shape=(M,M))
    return out+out.T


def matrixAppend(mat,vec):
    from scipy.sparse import coo_matrix
    nr,nc = mat.shape
    mat = mat.tocoo()
    r = mat.row.copy()
    c = mat.col.copy()
    d = mat.data.copy()
    if vec.ndim==1:
        if vec.size!=nr:
            print 'Array is not aligned with matrix'
            df
        r = numpy.concatenate((r,numpy.arange(vec.size)))
        c = numpy.concatenate((c,numpy.ones(vec.size)*nc))
        d = numpy.concatenate((d,vec))
        return coo_matrix((d,(r,c)),(nr,nc+1))
    if vec.shape[1]!=nr:
        print 'Array is not aligned with matrix'
        df
    import indexTricks as iT
    vr,vc = iT.coords(vec.shape)
    vc += nc
    r = numpy.concatenate((r,vr.ravel()))
    c = numpy.concatenate((c,vc.ravel()))
    d = numpy.concatenate((d,vec.ravel()))
    return coo_matrix((d,(r,c)),(nr,nc+vec.shape[0]))


def matrixCat(mat1,mat2):
    from scipy.sparse import coo_matrix
    nr,nc = mat1.shape
    nr2,nc2 = mat2.shape
    if nr!=nr2:
        print 'Matrices not aligned'
        df
    mat1 = mat1.tocoo()
    mat2 = mat2.tocoo()
    r = numpy.concatenate((mat1.row,mat2.row))
    c = numpy.concatenate((mat1.col,mat2.col+nc))
    d = numpy.concatenate((mat1.data,mat2.data))
    return coo_matrix((d,(r,c)),(nr,nc+nc2))


def matrixStack(mat1,mat2):
    from scipy.sparse import coo_matrix
    nr,nc = mat1.shape
    nr2,nc2 = mat2.shape
    if nc!=nc2:
        print 'Matrices not aligned!!!!'
    mat1 = mat1.tocoo()
    mat2 = mat2.tocoo()
    r = numpy.concatenate((mat1.row,mat2.row+nr))
    c = numpy.concatenate((mat1.col,mat2.col))
    d = numpy.concatenate((mat1.data,mat2.data))
    return coo_matrix((d,(r,c)),(nr+nr2,nc))


def getModelG(img,var,mat,cmat,rmat=None,reg=None,niter=10):
    from scikits.sparse.cholmod import cholesky
    from scipy.sparse import hstack
    import numpy
    rhs = mat.T*(img/var)

    B = fastMult(mat.T,cmat*mat)

    if rmat is not None:
        lhs = B+rmat*reg
        regs = [reg]
    else:
        niter = 0
        lhs = B
        reg = 0

    i = 0
    import time
    t = time.time()
    F = cholesky(lhs)
    print time.time()-t
    t = time.time()
    fit = F(rhs)
    print time.time()-t
    res = 0.
    for i in range(niter):
        res = fit.dot(rmat*fit)

        delta = reg*1e-3
        lhs2 = B+(reg+delta)*rmat

        T = (2./delta)*(numpy.log(F.cholesky(lhs2).L().diagonal()).sum()-numpy.log(F.L().diagonal()).sum())
        reg = abs(mat.shape[1]-T*reg)/res

        if abs(reg-regs[-1])/reg<0.005:
            break
        regs.append(reg)
        lhs = B+regs[-1]*rmat
        F = F.cholesky(lhs)
        fit = F(rhs)
        res = -0.5*res*regs[-1]
    res += -0.5*((mat*fit-img)**2/var).sum()

    return res,fit,mat*fit,rhs,(reg,i)


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

        delta = reg*1e-3
        lhs2 = B+(reg+delta)*rmat

        T = (2./delta)*(numpy.log(F.cholesky(lhs2).L().diagonal()).sum()-numpy.log(F.L().diagonal()).sum())
        reg = (omat.shape[1]-T*reg)/res
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


def getMatBilinear(xin,yin,srcx,srcy,srcxaxis,srcyaxis,mask=None):
    import numpy
    from scipy.sparse import coo_matrix

    if mask is None:
        mask = xin==xin

    size = xin.size
    scale = srcxaxis[1]-srcxaxis[0]
    spix = PixelNumber(xin,yin,srcxaxis,srcyaxis,mode='NearestBottomLeft')
    c = (spix>-1)&mask
    row = numpy.arange(spix.size)[c]
    xin,yin = xin[c],yin[c]
    spix = spix[c].astype(numpy.int32)

    # Create row, col, value arrays
    r = numpy.empty(row.size*4)
    c = numpy.empty(row.size*4).astype(numpy.int32)
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

    ssize = srcx.size
    """
    tmp = srcx.copy().astype(numpy.int32)
    u = numpy.unique(c)
    ssize = u.size
    np = numpy.arange(ssize).astype(numpy.int32)
    tmp[u] = np
    c = tmp[c]
    """

    return coo_matrix((w,(r,c)),shape=(size,ssize))

