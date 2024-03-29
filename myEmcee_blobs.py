import numpy
import emcee



global hackpars
global hackcosts
global hackdets


def optFunc(x):
    logp = 0.
    bad = False
    nvars = len(hackpars)
    for varIndx in xrange(nvars):
        hackpars[varIndx].value = x[varIndx]
        try:
            logp += hackpars[varIndx].logp
        except:
            logp = -1e300
            bad = True
            return -1e300
    for cost in hackcosts:
        logp += cost.logp
    if logp<-1e300:
        return -1e300
    return logp


def optFunc_blobs(x):
    logp = 0.
    bad = False
    nvars = len(hackpars)
    for varIndx in xrange(nvars):
        hackpars[varIndx].value = x[varIndx]
        try:
            logp += hackpars[varIndx].logp
        except:
            logp = -1e300
            bad = True
            return -1e300,[None for i in hackdets]
    for cost in hackcosts:
        logp += cost.logp
    if logp<-1e300:
        return -1e300,[None for i in hackdets]
    return logp,[i.value for i in hackdets]


def logprior(x):
    logp = 0.
    nvars = len(hackpars)
    for varIndx in xrange(nvars):
        hackpars[varIndx].value = x[varIndx]
        try:
            logp += hackpars[varIndx].logp
        except:
            return -numpy.inf
    return logp

def loglikelihood(x):
    logp = 0.
    for cost in hackcosts:
        lp = cost.logp
        if lp<=-1e300:
            return -numpy.inf
        logp += lp
    return logp


class Emcee:
    def __init__(self,pars,cov_u=None,cov=None,cov_n=None,nwalkers=None,nthreads=2,initialPars=None):
        global hackpars,hackcosts,hackdets
        self.pars = []
        self.costs = []
        self.deterministics = []
        for par in pars:
            try:
                if par.observed==True:
                    self.costs.append(par)
                else: self.pars.append(par)
            except:
                self.deterministics.append(par)
        self.nvars = len(self.pars)
        self.ndim = self.nvars
        if cov is None:
            self.cov_n = cov_n
        else:
            self.cov_n = cov
        if cov_u is None:
            cov_u = cov*0.
        self.cov_u = cov_u
        if nwalkers==None:
            self.nwalkers = self.ndim*5
        else:
            self.nwalkers = nwalkers
        if self.nwalkers%2==1:
            self.nwalkers = self.nwalkers+1
        hackpars = self.pars
        hackcosts = self.costs
        hackdets = self.deterministics
        if len(hackdets)==0:
            self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,optFunc,threads=nthreads)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,optFunc_blobs,threads=nthreads)
        self.p0 = initialPars


    def setStart(self):
        vals = numpy.array([p.value for p in self.pars])
        self.p0 = [vals]
        set = 1
        while set<self.nwalkers:
            if self.cov_n.ndim==1:
                var_u = numpy.random.random(self.ndim)*self.cov_u
                var_n = numpy.random.randn(self.ndim)*self.cov_n
                trial = vals+var_u+var_n
            else:
                trial = vals+numpy.random.randn(self.ndim)*self.cov_n
            ok = True
            for t in range(trial.size):
                self.pars[t].value = trial[t]
                try:
                    l = self.pars[t].logp
                except:
                    ok = False
                    break
            if ok==True:
                set += 1
                self.p0.append(trial)

    def sample(self,niter):
        if len(hackdets)!=0:
            self.sample_blobs(niter)
            return
        if self.p0 is None:
            self.setStart()

        pos, prob, state = self.sampler.run_mcmc(self.p0, 1)
        opos,oprob,orstate = [],[],[]
        for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=niter):
            opos.append(pos.copy())
            oprob.append(prob.copy())
            #orstate.append(rstate)
            #print pos,prob,rstate
        self.pos = numpy.array(opos)
        self.prob = numpy.array(oprob)

    def sample_save(self,niter,filename):
        if len(hackdets)!=0:
            self.sample_blobs(niter)
            return
        if self.p0 is None:
            self.setStart()

        of = open(filename, "w")
        of.close()
        pos, prob, state = self.sampler.run_mcmc(self.p0, 1)
        opos,oprob,orstate = [],[],[]
        for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=niter):
            opos.append(pos.copy())
            oprob.append(prob.copy())
            of = open(filename, "a")
            sf = open(filename+'logp',"a")
            position = pos.copy()
            for k in range(position.shape[0]):
                of.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,position[k]))))
            sf.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,prob))))
            of.close()
            sf.close()
        self.pos = numpy.array(opos)
        self.prob = numpy.array(oprob)

    def sample_blobs(self,niter):
        if self.p0 is None:
            self.setStart()
        #else:
        #    blobs = [None for i in self.p0]
        
        pos, prob, state, blobs = self.sampler.run_mcmc(self.p0, 1)
        opos,oprob,orstate,oblobs = [],[],[],[]
        for pos,prob,rstate,blobs in self.sampler.sample(pos,prob,state,iterations=niter,blobs0=blobs):
            opos.append(pos.copy())
            oprob.append(prob.copy())
            oblobs.append([b for b in blobs])
            #orstate.append(rstate)
            #print pos,prob,rstate
        self.pos = numpy.array(opos)
        self.prob = numpy.array(oprob)
        self.blobs = oblobs

    def result(self):
        nstep,nwalk = numpy.unravel_index(self.prob.argmax(),self.prob.shape)
        result = {}
        for i in range(self.nvars):
            result[self.pars[i].__name__] = self.pos[:,:,i]
        return self.prob,self.pos,result,self.pos[nstep,nwalk]


class PTEmcee:
    def __init__(self,pars,cov_u=None,cov=None,cov_n=None,nwalkers=None,ntemps=None,nthreads=2,initialPars=None,filename=None):
        global hackpars,hackcosts
        self.pars = []
        self.costs = []
        self.deterministics = []
        for par in pars:
            try:
                if par.observed==True:
                    self.costs.append(par)
                else: self.pars.append(par)
            except:
                self.deterministics.append(par)
        hackpars = self.pars
        hackcosts = self.costs

        self.nvars = len(self.pars)
        self.ndim = self.nvars
        if cov is None:
            self.cov_n = cov_n
        else:
            self.cov_n = cov
        if cov_u is None:
            cov_u = cov*0.
        self.cov_u = cov_u
        if nwalkers==None:
            self.nwalkers = self.ndim*5
        else:
            self.nwalkers = nwalkers
#        if self.nwalkers%2==1:
#            self.nwalkers = self.nwalkers+1
        if ntemps==None:
            self.ntemps = self.nwalkers
        else:
            self.ntemps = ntemps

        self.sampler = emcee.PTSampler(self.ntemps,self.nwalkers,self.ndim,loglikelihood,logprior,threads=nthreads)
        self.p0 = initialPars
        self.filename = filename

    def setStart(self):
        vals = numpy.array([p.value for p in self.pars])
        self.p0 = []
        for i in range(self.ntemps):
            p0 = [vals]
            set = 1
            while set<self.nwalkers:
                if self.cov_n.ndim==1:
                    var_u = numpy.random.random(self.ndim)*self.cov_u
                    var_n = numpy.random.randn(self.ndim)*self.cov_n
                    trial = vals+var_u+var_n
                else:
                    trial = numpy.random.multivariate_normal(vals,self.cov_n)
                ok = True
                for t in range(trial.size):
                    self.pars[t].value = trial[t]
                    try:
                        l = self.pars[t].logp
                    except:
                        ok = False
                        break
                if ok==True:
                    set += 1
                    p0.append(trial)
	    #print len(p0)
            self.p0.append(p0)
	    #print len(self.p0)
        #self.p0 = numpy.array(p0)
	self.p0 = numpy.array(self.p0)


    def sample(self,niter,thin=1,noTemps=False):
        if self.p0 is None:
            self.setStart()

        opos,ologp,ologl = [],[],[]
        if noTemps is False:
            opos = numpy.empty((niter/thin,self.ntemps,self.nwalkers,self.nvars))
            ologp = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
            ologl = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
        else:
            opos = numpy.empty((niter/thin,1,self.nwalkers,self.nvars))
            ologp = numpy.empty((niter/thin,1,self.nwalkers))
            ologl = numpy.empty((niter/thin,1,self.nwalkers))

        count = 0
        tloop = 0
        for pos,logp,logl in self.sampler.sample(self.p0,iterations=niter,storechain=False):
            if tloop%thin==0:
                if noTemps is False:
                    opos[count] = pos
                    ologp[count] = logp
                    ologl[count] = logl
                else:
                    opos[count] = pos[0]
                    ologp[count] = logp[0]
                    ologl[count] = logl[0]
                count += 1
            tloop += 1
        self.pos = numpy.array(opos)
        self.logpost = numpy.array(ologp)
        self.logl = numpy.array(ologl)
        #self.sampler.pool.close()
        #self.sampler.pool.join()

    ''' attempt at blobbing -- I don't think ptemcee currently does this...'''
    '''def sample_blobs(self,niter,thin=1,noTemps=False):
        if self.p0 is None:
            self.setStart()
            print 'set start'
        #else:
        blobs = [None for i in self.p0]
        
        opos,ologp,ologl,oblobs = [],[],[], []
        if noTemps is False:
            opos = numpy.empty((niter/thin,self.ntemps,self.nwalkers,self.nvars))
            ologp = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
            ologl = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
            oblobs = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
        else:
            opos = numpy.empty((niter/thin,1,self.nwalkers,self.nvars))
            ologp = numpy.empty((niter/thin,1,self.nwalkers))
            ologl = numpy.empty((niter/thin,1,self.nwalkers))
            oblobs = numpy.empty((niter/thin,1,self.nwalkers))

        count = 0
        tloop = 0
        for pos,logp,logl,blobs in self.sampler.sample(self.p0,iterations=niter,storechain=False,blobs0=blobs):
            if tloop%thin==0:
                if noTemps is False:
                    opos[count] = pos
                    ologp[count] = logp
                    ologl[count] = logl
                    oblobs[count] = blobs # only one type of blob
                else:
                    opos[count] = pos[0]
                    ologp[count] = logp[0]
                    ologl[count] = logl[0]
                    oblobs[count] = blobs[0] # may not work
                count += 1
            tloop += 1
        self.pos = numpy.array(opos)
        self.logpost = numpy.array(ologp)
        self.logl = numpy.array(ologl)
        self.blobs = numpy.array(blobs)'''

    def sample_save(self,niter,thin=1,noTemps=False):
        if self.p0 is None:
            self.setStart()

        of = open(self.filename, "w")
        of.close()

        opos,ologp,ologl = [],[],[]
        if noTemps is False:
            opos = numpy.empty((niter/thin,self.ntemps,self.nwalkers,self.nvars))
            ologp = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
            ologl = numpy.empty((niter/thin,self.ntemps,self.nwalkers))
        else:
            opos = numpy.empty((niter/thin,1,self.nwalkers,self.nvars))
            ologp = numpy.empty((niter/thin,1,self.nwalkers))
            ologl = numpy.empty((niter/thin,1,self.nwalkers))

        count = 0
        tloop = 0
        for pos,logp,logl in self.sampler.sample(self.p0,iterations=niter,storechain=False):
            if tloop%thin==0:
                if noTemps is False:
                    opos[count] = pos
                    ologp[count] = logp
                    ologl[count] = logl
                else:
                    opos[count] = pos[0]
                    ologp[count] = logp[0]
                    ologl[count] = logl[0]
                position = pos[0]
                of = open(self.filename, "a")
                for k in range(position.shape[0]):
                    of.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,position[k]))))
                of.close()
                count += 1
            tloop += 1
        self.pos = numpy.array(opos)
        self.logpost = numpy.array(ologp)
        self.logl = numpy.array(ologl)


        



    def result(self):
        shp = (self.logpost.shape[0],self.nwalkers)
        nstep,nwalk = numpy.unravel_index(self.logpost[:,0].argmax(),shp)
        result = {}
        for i in range(self.nvars):
            result[self.pars[i].__name__] = self.pos[:,:,:,i]
        #for i in range(len(self.deterministics)):
        #    d = []
        return self.logpost,self.pos,result,self.pos[nstep,0,nwalk]








