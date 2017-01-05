import scipy,numpy,cPickle
import special_functions as sf
from scipy import ndimage,optimize,signal,interpolate
from numpy import linalg
from math import sqrt,log,log10
import pymc, myEmcee_blobs as myEmcee
import numpy as np, pylab as pl
''' not the main one -- that's stitchfitter2! '''

light = 299792.458

def finddispersion(scispec,varspec,t1,t2,twave1,twave2,outwave,z,nfit=6,outfile=None,mask=None,bias=1e8,lim=4000.,bg='polynomial',restmask=None,lenslim=5500.):
    outwave,scispec,varspec = outwave[outwave>log10(lim)], scispec[outwave>log10(lim)], varspec[outwave>log10(lim)]
    print bias

    if mask is not None:
        ma = np.ones(outwave.size)
        for M in mask:
            cond = np.where((outwave>np.log10(M[0]))&(outwave<np.log10(M[1])))
            ma[cond]=0
        mask=ma==1
        outwave,scispec,varspec=outwave[mask],scispec[mask],varspec[mask]
    isig = 1./varspec**0.5
    ntemps1,ntemps2 = len(t1), len(t2)
    print ntemps1,ntemps2,nfit

    vL = pymc.Uniform('lens velocity',-3050.,3050.,value=0.)#,value=2200.)
    sL = pymc.Uniform('lens dispersion',5.,301.)
    pars = [vL,sL]
    cov = np.array([50.,10.])

    # Create the polynomial fit components
    BIAS = scispec*0.
    grid = 10**outwave[-1] - 10**outwave[0]
    operator = scipy.zeros((scispec.size,ntemps1+ntemps2+nfit))

    print nfit, outwave
    for i in range(nfit):
        p = scipy.zeros((nfit,1))
        p[i] = 1.
        coeff = {'coeff':p,'type':bg}
        poly = sf.genfunc(10**outwave,0.,coeff)
        operator[:,i+ntemps1+ntemps2] = poly
        print i+ntemps1+ntemps2, operator[:,i+ntemps1+ntemps2]
        BIAS += poly*bias/grid**i

    oper = operator.T  
    lenscond = np.where(outwave<np.log10(lenslim),True,False)
    @pymc.deterministic
    def logprob(value=0.,pars=pars):
        zL = np.log10(1.+z+vL.value/light)
        for k in range(ntemps1):
            oper[k,~lenscond] = interpolate.bisplev(sL.value,outwave[~lenscond]-zL,t1[k])
            #print k, oper[k]
        for k in range(ntemps2):
            oper[k+ntemps1,lenscond] = interpolate.bisplev(sL.value,outwave[lenscond]-zL,t2[k])
            #print k+ntemps1, oper[k+ntemps1]
        op = (oper*isig).T
        rhs = (scispec+BIAS)*isig
        fit,chi = optimize.nnls(op,rhs)
        lp = -0.5*chi**2.
        return lp
    
    @pymc.observed
    def logp(value=0.,lp=logprob):
        return lp
    print outfile
    S = myEmcee.Emcee(pars+[logp],cov=cov,nthreads=8,nwalkers=100)
    S.sample(900)
    outFile = '/data/ljo31b/EELs/esi/kinematics/inference/'+outfile
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()
    lp,trace,dic,_ = S.result()
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,i]
        print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

    velL,sigL = trace[a1,a2]
    
    
    zL = np.log10(1.+z+velL/light)
    for k in range(ntemps1):
        oper[k,~lenscond] = interpolate.bisplev(sigL,outwave[~lenscond]-zL,t1[k])
    for k in range(ntemps2):
        oper[k+ntemps1,lenscond] = interpolate.bisplev(sigL,outwave[lenscond]-zL,t2[k])

    op = (oper*isig).T
    rhs = (scispec+BIAS)*isig
    fit,chi = optimize.nnls(op,rhs)
    for i in range(nfit):
        fit[ntemps1+ntemps2+i] -= bias/grid**i
    outmodel = scipy.dot(oper.T,fit)
    continuum = scipy.dot(operator,fit)

    pl.figure()
    pl.plot(10**outwave,scispec)
    pl.plot(10**outwave,outmodel)
    pl.plot(10**outwave,continuum)
    pl.figure()
    pl.plot(lp)
    pl.show()
    return S.result()

def readresults(scispec,varspec,t1,t2,twave1,twave2,outwave,z,nfit=6,infile=None,mask=None,bias=1e8,lim=5000.,bg='polynomial',restmask=None,lenslim=5500.):
    outwave,scispec,varspec = outwave[outwave>log10(lim)], scispec[outwave>log10(lim)], varspec[outwave>log10(lim)]
    origwave,origsci,origvar = outwave.copy(),scispec.copy(),varspec.copy()
    if mask is not None:
        ma = np.ones(outwave.size)
        for M in mask:
            cond = np.where((outwave>np.log10(M[0]))&(outwave<np.log10(M[1])))
            ma[cond]=0
        if restmask is not None:
            for M in restmask:
                cond = np.where((outwave>np.log10(M[0]*(1.+zl)))&(outwave<np.log10(M[1]*(1.+zl))))
                ma[cond]=0
        ma=ma==1
        outwave,scispec,varspec=outwave[ma],scispec[ma],varspec[ma]
    isig = 1./varspec**0.5
    ntemps1,ntemps2 = len(t1), len(t2)
    print ntemps1,ntemps2

   # Create the polynomial fit components
    BIAS = scispec*0.
    grid = 10**outwave[-1] - 10**outwave[0]
    operator = scipy.zeros((scispec.size,ntemps1+ntemps2+nfit))
    for i in range(nfit):
        p = scipy.zeros((nfit,1))
        p[i] = 1.
        coeff = {'coeff':p,'type':bg}
        poly = sf.genfunc(10**outwave,0.,coeff)
        operator[:,i+ntemps1+ntemps2] = poly
        BIAS += bias*poly/grid**i

    oper = operator.T 
    lenscond = np.where(outwave<np.log10(lenslim),True,False)

    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+infile)
    lp,trace,dic,_=result
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    velL,sigL= trace[a1,a2]

    zL = np.log10(1.+z+velL/light)
    for k in range(ntemps1):
        oper[k,~lenscond] = interpolate.bisplev(sigL,outwave[~lenscond]-zL,t1[k])
    for k in range(ntemps2):
        oper[k+ntemps1,lenscond] = interpolate.bisplev(sigL,outwave[lenscond]-zL,t2[k])
        
    op = (oper*isig).T
    rhs = (scispec+BIAS)*isig
    fit,chi = optimize.nnls(op,rhs)
    for i in range(nfit):
        fit[ntemps1+ntemps2+i] -= bias/grid**i
    maskmodel = scipy.dot(oper.T,fit)
    # unmasked
    if mask is not None or restmask is not None:
        operator = scipy.zeros((origsci.size,ntemps1+ntemps2+nfit))
        
        for i in range(nfit):
            p = scipy.zeros((nfit,1))
            p[i] = 1.
            coeff = {'coeff':p,'type':bg}
            poly = sf.genfunc(10**origwave,0.,coeff)
            operator[:,i+ntemps1+ntemps2] = poly

        oper = operator.T 
        origlenscond = np.where(origwave<np.log10(lenslim),True,False)

        for k in range(ntemps1):
            oper[k,~origlenscond] = interpolate.bisplev(sigL,origwave[~origlenscond]-zL,t1[k])
        for k in range(ntemps2):
            oper[k+ntemps1,origlenscond] = interpolate.bisplev(sigL,origwave[origlenscond]-zL,t2[k])
      
    outmodel = scipy.dot(oper.T,fit)
    lens = scipy.dot(oper[:ntemps1].T,fit[:ntemps1]) + scipy.dot(oper[ntemps1:ntemps1+ntemps2].T,fit[ntemps1:ntemps1+ntemps2])
    cont = scipy.dot(oper[ntemps1+ntemps2:].T,fit[ntemps1+ntemps2:])
    
    pl.figure()
    pl.subplot(211)
    #print restmask
    if mask is not None:
        for M in mask:
            pl.axvspan(M[0], M[1], color='DarkGray')
    if restmask is not None:
        for M in restmask:
            pl.axvspan(M[0]*(1.+zl),M[1]*(1.+zl),color='DarkGray')
            pl.axvspan(M[0]*(1.+zs), M[1]*(1.+zs),color='DarkGray')
    pl.plot(10**origwave,origsci,'LightGray',)
    pl.plot(10**origwave,outmodel,'k',)
    pl.plot(10**origwave,lens,'SteelBlue')
    pl.plot(10**origwave,cont,'Navy')
    pl.legend(loc='upper right',frameon=False)
    #pl.xlabel('observed wavelength ($\AA$)')
    pl.ylabel('flux')
    pl.axis([lim,9000,-0.5,4])
    pl.figtext(0.15,0.85,r'$\sigma_{l} = $'+'%.2f'%sigL)
    pl.subplot(212)
    if mask is not None:
        for M in mask:
            pl.axvspan(M[0], M[1], color='DarkGray')
    if restmask is not None:
        for M in restmask:
            pl.axvspan(M[0]*(1.+zl),M[1]*(1.+zl),color='DarkGray')
            pl.axvspan(M[0]*(1.+zs), M[1]*(1.+zs),color='DarkGray')
    pl.plot(10**origwave,(origsci-outmodel)/origvar,'k')
    pl.xlabel('observed wavelength ($\AA$)')
    pl.ylabel('residuals')
    pl.axis([lim,8500,-120,120])
    #pl.show()
    #pl.figure()
    #pl.plot(10**outwave,scispec-maskmodel,'k')
    print '%.2f'%velL, '&','%.2f'%sigL
    for i in fit:
        print i
    return result

