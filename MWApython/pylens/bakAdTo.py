import numpy

class AdaptiveSource:
    def __init__(self,pix,gridsize,useSN=False):
        if useSN is False:
            self.indxpix = numpy.random.random(pix.size).argsort()[:gridsize]
        else:
            self.indxpix = self.drawPixelsSN(pix,gridsize)
        self.gridsize = gridsize
        self.grid = None
        self.pnts = None
        self.gridpnts = None
        self.lmat = None
        self.rmat = None


    def drawPixelsSN(self,pix,gridsize):
        from scipy import interpolate
        apix = pix.argsort()
        spix = numpy.sort(pix)
        spix -= spix.min()
        cpix = spix.cumsum()
        cpix -= cpix[0]
        cpix /= cpix[-1]

        model = interpolate.splrep(cpix,numpy.arange(cpix.size),s=1)
        A = numpy.unique(interpolate.splev(numpy.random.random(gridsize,model).astype(numpy.int32)))
        while A.size<gridsize:
            A = numpy.unique(numpy.concatenate((A,interpolate.splev(numpy.random.random(gridsize,model).astype(numpy.int32)))))
        if A.size>gridsize:
            A = A[numpy.random.random(A.size).argsort()[:gridsize]]
        return apix[A]


    def update(self,x,y):
        from scipy import spatial
        self.pnts = numpy.array([x,y]).T
        self.gridpnts = self.pnts[self.indxpix]
        self.grid = spatial.Delaunay(self.gridpnts)

        self._setLensMatrix()
        self._setCurvatureRegularization()


    def _setLensMatrix(self):
        from scipy.sparse import coo_matrix

        x,y = self.pnts.T
        srcPix = self.grid.find_simplex(self.pnts)
        gridVerts = self.grid.simplices[srcPix]

        (x1,x2,x3),(y1,y2,y3) = self.gridpnts[gridVerts].T

        bot = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
        wa = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/bot
        wb = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/bot
        wc = 1.-wb-wa

        W = [wa,wb,wc]
        row = numpy.arange(x.size)
        r = numpy.empty(row.size*3)
        c = numpy.empty(row.size*3)
        v = numpy.empty(row.size*3)

        cond = (wa>=0)&(wb>=0)&(wc>=0)&(wa<=1.)&(wb<=1.)&(wc<=1.)
        cond *= 1.
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

        dcq = gx-xn
        dab = ((x1-x2)**2+(y1-y2)**2)**0.5
        dqa = ((x1-xn)**2+(y1-y3)**2)**0.5
        dqb = ((x2-xn)**2+(y2-y3)**2)**0.5

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

        dcp = gx+xp
        dde = ((x1-x2)**2+(y1-y2)**2)**0.5
        dpd = ((x1-xp)**2+(y1-y3)**2)**0.5
        dpe = ((x2-xp)**2+(y2-y3)**2)**0.5

        # Create the sx-curvature matrix
        wa = dqb/(dcq*dab)
        wb = dqa/(dcq*dab)
        wc = -(1./dcp+1./dcq)
        wd = dpe/(dcp*dde)
        we = dpd/(dcp*dde)

        a,b,c = nidx
        d,e,C = pidx

        n = OK.sum()
        r = numpy.empty(n*5)
        c = r*0
        v = r*0
        W = [wa,wb,wc,wd,we]
        C = [a,b,C,d,e]
        for i in range(5):
            S = slice(i*n,(i+1)*n,1)
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
        OK = (IND.sum(1)==1)#&OK
        yn = y1+(x3-x1)*(y2-y1)/(x2-x1)

        dcq = gy-yn
        dab = ((x1-x2)**2+(y1-y2)**2)**0.5
        dqa = ((x1-x3)**2+(y1-yn)**2)**0.5
        dqb = ((x2-x3)**2+(y2-yn)**2)**0.5

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
        OK = (IND.sum(1)==1)&OK
        yp = y1+(x3-x1)*(y2-y1)/(x2-x1)

        dcp = gy+yp
        dde = ((x1-x2)**2+(y1-y2)**2)**0.5
        dpd = ((x1-x3)**2+(y1-yp)**2)**0.5
        dpe = ((x2-x3)**2+(y2-yp)**2)**0.5

        # Create the sy-curvature matrix
        wa = dqb/(dcq*dab)
        wb = dqa/(dcq*dab)
        wc = -(1./dcp+1./dcq)
        wd = dpe/(dcp*dde)
        we = dpd/(dcp*dde)

        a,b,c = nidx
        d,e,C = pidx

        n = OK.sum()
        r = numpy.empty(n*5)
        c = r*0
        v = r*0
        W = [wa,wb,wc,wd,we]
        C = [a,b,C,d,e]
        for i in range(5):
            S = slice(i*n,(i+1)*n,1)
            r[S] = A[OK]
            c[S] = C[i][OK]
            v[S] = W[i][OK]
        sy = coo_matrix((v,(r,c)),shape=(A.size,A.size))

        self.rmat = sx.T*sx+sy.T*sy


    def evaluate(self,x,y,vals):
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
        src[~C] = numpy.nan

        return src

    def eval(self,x,y,vals):
        return self.evaluate(x,y,vals)


def fastAAT(a):
    import numpy
    import matmult

    b = a.tocsr()
    a = a.T.tocsr()

    M,N = a.shape
    b.sort_indices()

    iw = numpy.empty(max(a.shape),dtype=numpy.intc)
    nnz = matmult.count_t(M,a.indices+1,a.indptr+1,b.indices+1,b.indptr+1,iw)
    c,jc,ic = matmult.multd_t(M,M,a.indices,a.indptr,b.indices,b.indptr,a.data,b.data,nnz)
    out = a.__class__((c,jc,ic),shape=(M,M))
    return out+out.T


def getModelG(img,var,mat,cmat,rmat=None,reg=None,niter=10):
    from scikits.sparse.cholmod import cholesky
    from scipy.sparse import hstack
    import numpy
    rhs = mat.T*(img/var)

    B = fastAAT(cmat*mat)

    if rmat is not None:
        lhs = B+rmat*reg
        regs = [reg]
    else:
        niter = 0
        lhs = B
        reg = 0

    i = 0
    F = cholesky(lhs)
    fit = F(rhs)
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

