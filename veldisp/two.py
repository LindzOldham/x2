    def getSpectra(self,wave,z,disp):
        wrest = wave/(1.+z)
        wcond = (self.wave>wrest.min()-self.nsigma*self.pixScale*2)
        wcond = wcond&(self.wave<wrest.max()+self.nsigma*self.pixScale*2)

        w = self.wave[wcond]
        X = numpy.arange(w.size)
        kernel = w*(disp**2+self.sigma2diff[wcond])**0.5/299792.

        kpix = kernel/self.pixScale
        bcond = ~numpy.isfinite(kpix)|(kpix<0.5)
        kpix[bcond] = 0
        kpix2 = kpix**2
        pmax = int(kpix.max()*self.nsigma+1)

        rcol = numpy.linspace(-pmax,pmax,2*pmax+1).repeat(X.size).reshape((2*pmax+1,X.size))
        col = rcol+X
        row = X.repeat(2*pmax+1).reshape((X.size,2*pmax+1)).T
        c = (col>=0)&(col<w.size)&(abs(rcol)<=self.nsigma*kpix[row])
        col = col[c]
        row = row[c]
        rcol = rcol[c]
        kpix2 = kpix2[row]
        wht = numpy.exp(-0.5*rcol*rcol/kpix2)/(self.norm*kpix[row])
        wht[bcond[row]] = 1.
        B = sparse.coo_matrix((wht,(row,col)),shape=(X.size,X.size))

        indx = (wrest-w[0])/(w[1]-w[0])
        lo = indx.astype(numpy.int32)
        hi = lo+1
        weights = 1.-numpy.concatenate((indx-lo,hi-indx))
        cols = numpy.concatenate((lo,hi))
        rows = numpy.concatenate((numpy.arange(lo.size),numpy.arange(lo.size)))
        cond = (cols<0)|(cols>=w.size)
        cols[cond] = 0
        weights[cond] = 0.
        
        I = sparse.coo_matrix((weights,(rows,cols)),shape=(lo.size,X.size))
        op = I*B

        ospex = []
        for spec in self.spex:
            ospex.append(op*spec[wcond])

        return ospex
