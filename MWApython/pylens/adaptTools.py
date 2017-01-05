import numpy

class AdaptiveSource:
    def __init__(self,pix,gridsize,drawRandom=False,useSN=False,scaleSN=1.):
        if drawRandom==True:
            self.indxpix = numpy.random.random(pix.size).argsort()[:gridsize]
        elif useSN==True:
            self.indxpix = self.drawPixelsSN(pix,gridsize,scaleSN)
        else:
            i = int(pix.size/gridsize)
            self.indxpix = numpy.arange(pix.size)[::i]
            gridsize = self.indxpix.size

        self.gridsize = gridsize
        self.insize = gridsize
        self.grid = None
        self.pnts = None
        self.gridpnts = None
        self.lmat = None
        self.rmat = None


    def drawPixelsSN(self,pix,gridsize,scaling):
        from scipy import interpolate
        apix = pix.argsort()
        spix = numpy.sort(pix)
        spix -= spix.min()
        cpix = spix.cumsum()
        cpix -= cpix[0]
        cpix /= cpix[-1]
        cpix = cpix**scaling

        model = interpolate.splrep(cpix,numpy.arange(cpix.size),s=1)
        A = numpy.unique(interpolate.splev(numpy.random.random(gridsize),model).astype(numpy.int32))
        while A.size<gridsize:
            A = numpy.unique(numpy.concatenate((A,interpolate.splev(numpy.random.random(gridsize),model).astype(numpy.int32))))
        if A.size>gridsize:
            A = A[numpy.random.random(A.size).argsort()[:gridsize]]
        return apix[A]


    def update(self,x,y,doReg=True,robust=True,verbose=False):
        from scipy import spatial
        import time
        self.pnts = numpy.array([x,y]).T

        indx = self.indxpix
        if robust==True:
            cnvxhll = spatial.ConvexHull(self.pnts)
            indx = numpy.unique(numpy.concatenate((cnvxhll.vertices,indx)))
            if verbose==True:
                print "%d points added"%(indx.size-self.indxpix.size)

        self.usedpix = indx
        self.gridpnts = self.pnts[indx]
        self.grid = spatial.Delaunay(self.gridpnts)
        self.gridsize = indx.size

        self._setLensMatrix()
        if doReg==True:
            self._setCurvatureRegularization()

        return
#        return
        t = time.time()
        self.grid = spatial.Delaunay(self.gridpnts)
        print time.time()-t
        t0 = time.time()
        self.grid = spatial.Delaunay(self.gridpnts)
        inout = self.grid.find_simplex(self.pnts)
        newpnts = numpy.unique(numpy.concatenate((numpy.where(inout==-1)[0],self.indxpix)))
        tmp = spatial.Delaunay(self.pnts[newpnts])
        newindx = numpy.unique(numpy.concatenate((newpnts[numpy.unique(tmp.convex_hull)],self.indxpix)))
        self.grid = spatial.Delaunay(self.pnts[newindx])
        print time.time()-t0,newindx
        t1 = time.time()
        nugrid = spatial.Delaunay(self.pnts)
        newindx = numpy.unique(numpy.concatenate((numpy.arange(x.size)[numpy.unique(nugrid.convex_hull)],self.indxpix)))
        self.grid = spatial.Delaunay(self.pnts[newindx])
        print time.time()-t1,newindx
        t2 = time.time()
        cv = spatial.ConvexHull(self.pnts)
        newindx = numpy.unique(numpy.concatenate((cv.vertices,self.indxpix)))
        self.grid = spatial.Delaunay(self.pnts[newindx])
        print time.time()-t2,newindx

        df



    def _setLensMatrix(self):
        from scipy.sparse import coo_matrix

        x,y = self.pnts.T
        srcPix = self.grid.find_simplex(self.pnts)
        cond = srcPix>-1
        gridVerts = self.grid.simplices[srcPix]

        (x1,x2,x3),(y1,y2,y3) = self.gridpnts[gridVerts].T

        bot = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
        wa = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/bot
        wb = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/bot
        wc = 1.-wb-wa

        W = [wa,wb,wc]
        row = numpy.arange(x.size)
        r = numpy.zeros(row.size*3)
        c = numpy.zeros(row.size*3)
        v = numpy.zeros(row.size*3)

        cond = cond*1.
        for i in range(3):
            S = slice(i*row.size,(i+1)*row.size,1)
            r[S] = row.copy()
            c[S] = gridVerts[:,i]
            v[S] = W[i]*cond

        self.lmat = coo_matrix((v,(r,c)),shape=(row.size,self.gridsize))


    def _setCurvatureRegularization(self):
        from scipy.sparse import coo_matrix

        gridpnts = self.gridpnts
        grid = self.grid
        gx,gy = gridpnts.T

        # Find negative-x pixels
        dxn = numpy.array([gx-1e-7,gy]).T
        gridVerts = grid.simplices[grid.find_simplex(dxn)]
        X,Y = gridpnts[gridVerts].T
        A = numpy.arange(self.gridsize)
        IND = (gridVerts.T==A).T
        idx = numpy.argsort(IND,1)
        x1,x2,x3 = X[idx,A[:,numpy.newaxis]].T
        y1,y2,y3 = Y[idx,A[:,numpy.newaxis]].T
        nidx = gridVerts.T[idx,A[:,numpy.newaxis]].T
        OKN = IND.sum(1)==1
        xn = x1+(y3-y1)*(x2-x1)/(y2-y1)

        dcq = abs(gx-xn)
        dab = ((x1-x2)**2+(y1-y2)**2)**0.5
        dqa = ((x1-xn)**2+(y1-y3)**2)**0.5
        dqb = ((x2-xn)**2+(y2-y3)**2)**0.5
        dcq[dcq==0] = 1e-11


        # Find positive-x pixels
        dxp = numpy.array([gx+1e-7,gy]).T
        gridVerts = grid.simplices[grid.find_simplex(dxp)]
        X,Y = gridpnts[gridVerts].T
        IND = (gridVerts.T==A).T
        idx = numpy.argsort(IND,1)
        x1,x2,x3 = X[idx,A[:,numpy.newaxis]].T
        y1,y2,y3 = Y[idx,A[:,numpy.newaxis]].T
        pidx = gridVerts.T[idx,A[:,numpy.newaxis]].T
        OKP = IND.sum(1)==1
        xp = x1+(y3-y1)*(x2-x1)/(y2-y1)

        dcp = abs(gx-xp)
        dde = ((x1-x2)**2+(y1-y2)**2)**0.5
        dpd = ((x1-xp)**2+(y1-y3)**2)**0.5
        dpe = ((x2-xp)**2+(y2-y3)**2)**0.5
        dcp[dcp==0] = 1e-11

        # Create the sx-curvature matrix
        wa = dqb/(dcq*dab)
        wb = dqa/(dcq*dab)
        wc = -(1./dcp+1./dcq)
        wd = dpe/(dcp*dde)
        we = dpd/(dcp*dde)

        wc1 = -1./dcq
        wc2 = -1./dcp

        a,b,cn = nidx
        d,e,cp = pidx


        OK = OKN&OKP
        n = OK.sum()
        N = n*5+3*((OKN|OKP).sum()-n)
        r = numpy.zeros(N)
        c = r*0
        v = r*0
        W = [wa,wb,wc,wd,we]
        C = [a,b,cn,d,e]
        for i in range(5):
            S = slice(i*n,(i+1)*n,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]
        OK = OKN&~OKP
        nn = OK.sum()
        W = [wa,wb,wc1]
        C = [a,b,cn]
        for i in range(3):
            S = slice(i*nn+n*5,(i+1)*nn+n*5,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]
        OK = OKP&~OKN
        np = OK.sum()
        W = [wd,we,wc2]
        C = [d,e,cp]
        for i in range(3):
            S = slice(i*np+n*5+3*nn,(i+1)*np+n*5+3*nn,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]

        sx = coo_matrix((v,(r,c)),shape=(A.size,A.size))


        # Find negative-y pixels
        dyn = numpy.array([gx,gy-1e-7]).T
        gridVerts = grid.simplices[grid.find_simplex(dyn)]
        X,Y = gridpnts[gridVerts].T
        IND = (gridVerts.T==A).T
        idx = numpy.argsort(IND,1)
        x1,x2,x3 = X[idx,A[:,numpy.newaxis]].T
        y1,y2,y3 = Y[idx,A[:,numpy.newaxis]].T
        nidx = gridVerts.T[idx,A[:,numpy.newaxis]].T
        OKN = IND.sum(1)==1
        yn = y1+(x3-x1)*(y2-y1)/(x2-x1)

        dcq = abs(gy-yn)
        dab = ((x1-x2)**2+(y1-y2)**2)**0.5
        dqa = ((x1-x3)**2+(y1-yn)**2)**0.5
        dqb = ((x2-x3)**2+(y2-yn)**2)**0.5
        dcq[dcq==0] = 1e-11

        # Find positive-y pixels
        dyp = numpy.array([gx,gy+1e-7]).T
        gridVerts = grid.simplices[grid.find_simplex(dyp)]
        X,Y = gridpnts[gridVerts].T
        #A = numpy.arange(I.size)
        IND = (gridVerts.T==A).T
        idx = numpy.argsort(IND,1)
        x1,x2,x3 = X[idx,A[:,numpy.newaxis]].T
        y1,y2,y3 = Y[idx,A[:,numpy.newaxis]].T
        pidx = gridVerts.T[idx,A[:,numpy.newaxis]].T
        OKP = IND.sum(1)==1
        yp = y1+(x3-x1)*(y2-y1)/(x2-x1)

        dcp = abs(gy-yp)
        dde = ((x1-x2)**2+(y1-y2)**2)**0.5
        dpd = ((x1-x3)**2+(y1-yp)**2)**0.5
        dpe = ((x2-x3)**2+(y2-yp)**2)**0.5
        dcp[dcp==0] = 1e-11

        # Create the sy-curvature matrix
        wa = dqb/(dcq*dab)
        wb = dqa/(dcq*dab)
        wc = -(1./dcp+1./dcq)
        wd = dpe/(dcp*dde)
        we = dpd/(dcp*dde)

        wc1 = -1./dcq
        wc2 = -1./dcp

        a,b,cn = nidx
        d,e,np = pidx

        OK = OKN&OKP
        n = OK.sum()
        N = n*5+3*((OKN|OKP).sum()-n)
        r = numpy.zeros(N)
        c = r*0
        v = r*0
        W = [wa,wb,wc,wd,we]
        C = [a,b,cn,d,e]
        for i in range(5):
            S = slice(i*n,(i+1)*n,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]
        OK = OKN&~OKP
        nn = OK.sum()
        W = [wa,wb,wc1]
        C = [a,b,cn]
        for i in range(3):
            S = slice(i*nn+n*5,(i+1)*nn+n*5,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]
        OK = OKP&~OKN
        np = OK.sum()
        W = [wd,we,wc2]
        C = [d,e,cp]
        for i in range(3):
            S = slice(i*np+n*5+3*nn,(i+1)*np+n*5+3*nn,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]

        sy = coo_matrix((v,(r,c)),shape=(A.size,A.size))

        self.rmat = sx.T*sx+sy.T*sy


    def evaluate(self,x,y,vals,domask=True):
        grid = self.grid
        pnts = numpy.array([x,y]).T

        srcPix = grid.find_simplex(pnts)
        gridVerts = grid.simplices[srcPix]

        (x1,x2,x3),(y1,y2,y3) = self.gridpnts[gridVerts].T

        bot = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
        wa = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/bot
        wb = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/bot
        wc = 1.-wb-wa
        C = (wa>=0)&(wb>=0)&(wc>=0)

        gV = gridVerts
        src = wa*vals[gV[:,0]]+wb*vals[gV[:,1]]+wc*vals[gV[:,2]]

        if domask:
            src[~C] = numpy.nan
            return src
        else:
            src[~C] = 0.
            return src,C

    def eval(self,x,y,vals,domask=True):
        return self.evaluate(x,y,vals,domask)


def fastAAT(a):
    import numpy
    import matmult

    b = a.tocsr()
    a = a.T.tocsr()

    M,N = a.shape
    b.sort_indices()

    iw = numpy.zeros(max(a.shape),dtype=numpy.intc)
    nnz = matmult.count_t(M,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    c,jc,ic = matmult.multd_t(M,M,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    out = a.__class__((c,jc,ic),shape=(M,M))
    return out+out.T


def getModelG(img,var,mat,cmat,rmat=None,reg=None,niter=10):
    from scikits.sparse.cholmod import cholesky
    from scipy.sparse import csc_matrix
    import numpy
    import pixellatedTools as pT
    rhs = mat.T*csc_matrix(img/var).T

    B = fastAAT(cmat*mat)
    #B = pT.fastMult(mat.T,cmat*mat)

    if rmat is not None and reg is not None:
        lhs = B+rmat*reg
        regs = [reg]
    else:
        niter = 0
        lhs = B
        reg = 0

    F = cholesky(lhs)
    fit = F(rhs).data.ravel()
    """
    from scipy.sparse import linalg
    print 'chol',fit

    fit = linalg.lsmr(lhs,rhs,atol=1e-13,btol=1e-13)[0]
    print 'smr',fit

    rhs = (img/var**0.5)
    lhs = cmat*mat
    fit = linalg.lsmr(lhs,rhs,atol=1e-13,btol=1e-13)[0]
    print 'smr2',fit
    fit = ffit
    """

    i = 0
    res = 0.
    if rmat is not None and reg>0:
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
        fit = F(rhs).data.ravel()
      res = -0.5*res*regs[-1]
    res += -0.5*((mat*fit-img)**2/var).sum()

    return res,fit,mat*fit,rhs,(reg,i)

