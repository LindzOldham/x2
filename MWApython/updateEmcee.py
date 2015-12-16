import numpy
import emcee

global hackpars
global hackcosts

def optFunc(x):
    logp = 0.
    bad = False
    nvars = len(hackpars)
    for varIndx in xrange(nvars):
        hackpars[varIndx].value = x[varIndx]
        try:
            logp += hackpars[varIndx].logp
        except:
            logp = -1e200
            bad = True
            break
    if bad==False:
        for cost in hackcosts:
            logp += cost.logp
    return logp


class Emcee:
    def __init__(self,pars,cov=None,nwalkers=None,nthreads=2):
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
        self.nvars = len(self.pars)
        self.ndim = self.nvars
        self.cov = cov
        if nwalkers==None:
            self.nwalkers = self.ndim*5
        else:
            self.nwalkers = nwalkers
        if self.nwalkers%2==1:
            self.nwalkers = self.nwalkers+1
        hackpars = self.pars
        hackcosts = self.costs
        self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,optFunc,threads=nthreads)

    def optFunc2(self,x):
        logp = 0.
        bad = False
        for varIndx in xrange(self.nvars):
            self.pars[varIndx].value = x[varIndx]
            try:
                logp += self.pars[varIndx].logp
            except:
                logp = -1e200
                bad = True
                break
        if bad==False:
            for cost in self.costs:
                logp += cost.logp
        return logp


    def sample(self,niter):
        vals = numpy.array([p.value for p in self.pars])
        p0 = [vals]
        set = 1
        while set<self.nwalkers:
            if self.cov.ndim==1:
                trial = vals+numpy.random.randn(self.ndim)*self.cov
            else:
                trial = numpy.random.multivariate_normal(vals,self.cov)
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

        pos, prob, state = self.sampler.run_mcmc(p0, 1)
        opos,oprob,orstate = [],[],[]
        for pos,prob,rstate in self.sampler.sample(pos,prob,state,iterations=niter):
            opos.append(pos.copy())
            oprob.append(prob.copy())
            #orstate.append(rstate)
            #print pos,prob,rstate
        self.pos = numpy.array(opos)
        self.prob = numpy.array(oprob)

    def result(self):
        nstep,nwalk = numpy.unravel_index(self.prob.argmax(),self.prob.shape)
        result = {}
        for i in range(self.nvars):
            result[self.pars[i].__name__] = self.pos[:,:,i]
        #for i in range(len(self.deterministics)):
        #    d = []
        return self.prob,self.pos,result,self.pos[nstep,nwalk]
