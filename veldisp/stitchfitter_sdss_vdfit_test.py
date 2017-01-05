import scipy,numpy,cPickle
import special_functions as sf
from scipy import ndimage,optimize,signal,interpolate
from numpy import linalg
from math import sqrt,log,log10
import pymc, myEmcee_blobs as myEmcee
import numpy as np, pylab as pl
import VDfit
from SampleOpt import AMAOpt,Sampler

''' bc03 and miles models '''

light = 299792.458

def finddispersion(scispec,varspec,t1,t2,outwave,zl,zs,nfit=12,outfile=None,mask=None,bias=1e8,lim=4000.,bg='polynomial',restmask=None,srclim=6000.,maxlim=8000.,nthr=8,name=None):
    outwave,scispec,varspec = outwave[outwave>log10(lim)], scispec[outwave>log10(lim)], varspec[outwave>log10(lim)]
    print 'BIAS', np.log10(bias)

    if mask is not None:
        ma = np.ones(outwave.size)
        for M in mask:
            cond = np.where((outwave>np.log10(M[0]))&(outwave<np.log10(M[1])))
            ma[cond]=0
        if restmask is not None:
            for M in restmask:
                cond = np.where((outwave>np.log10(M[0]*(1.+zl)))&(outwave<np.log10(M[1]*(1.+zl))))
                ma[cond]=0
                cond = np.where((outwave>np.log10(M[0]*(1.+zs)))&(outwave<np.log10(M[1]*(1.+zs))))
                ma[cond]=0
        mask=ma==1
        outwave,scispec,varspec=outwave[mask],scispec[mask],varspec[mask]
    isig = 1./varspec**0.5
    ntemps1,ntemps2 = t1.nspex, t2.nspex
    
    if name is None:
        vL = pymc.Uniform('lens velocity',-1050.,1050.)
        sL = pymc.Uniform('lens dispersion',20.,501.)
        vS = pymc.Uniform('source velocity',-1050.,1050.)
        sS = pymc.Uniform('source dispersion',20.,501.)
    else:
        print 'we are looking at EEL',name
        result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+name+'_1.00_lens_esi_indous_vdfit')
        lp,trace,dic,_=result
        a1,a2=np.unravel_index(lp.argmax(),lp.shape)
        vL = pymc.Uniform('lens velocity',-1050.,1050.,value=dic['lens velocity'][a1,a2])
        sL = pymc.Uniform('lens dispersion',5.,501.,value=dic['lens dispersion'][a1,a2])
        vS = pymc.Uniform('source velocity',-1050.,1050.,value=dic['source velocity'][a1,a2])
        sS = pymc.Uniform('source dispersion',5.,501.,value=dic['source dispersion'][a1,a2])
    pars = [vL,sL,vS,sS]
    cov = np.array([50.,10.,50.,10.])

    # Create the polynomial fit components
    BIAS = scispec*0.
    #grid = 10**outwave[-1] - 10**outwave[0]
    operator = scipy.zeros((scispec.size,2*ntemps1+ntemps2+nfit))
    X = np.arange(outwave.size)
    ff = 2.*X/X.max() - 1.
    for i in range(nfit):
        p = scipy.zeros((nfit,1))
        p[i] = 1.
        coeff = {'coeff':p,'type':bg}
        poly = sf.genfunc(ff,0.,coeff)
        operator[:,i+2*ntemps1+ntemps2] = poly
        BIAS += poly*bias#/grid**i
        
    oper = operator.T  
    cond = np.where(outwave<=np.log10(srclim),True,False)
    linwave=10**outwave
    @pymc.deterministic
    def logprob(value=0.,pars=pars):
        zL, zS = zl+vL.value/light, zs+vS.value/light
        oper[:ntemps1] = t1.getSpectra(linwave,zL,sL.value)
        oper[ntemps1:2.*ntemps1,~cond] = t1.getSpectra(linwave[~cond],zS,sS.value)
        oper[2*ntemps1:2*ntemps1+ntemps2,cond] = t2.getSpectra(linwave[cond],zS,sS.value)
        op = (oper*isig).T
        rhs = (scispec+BIAS)*isig
        fit,chi = optimize.nnls(op,rhs)
        lp = -0.5*chi**2.
        return lp
    
    @pymc.observed
    def logp(value=0.,lp=logprob):
        return lp
    print 'filename: ', outfile

    # optimise first
    SS = AMAOpt(pars,[logp],[logprob],cov=cov)
    SS.sample(1000)
    lp,trace,det = SS.result()
    
    print 'results from optimisation:'
    for i in range(len(pars)):
        pars[i].value = trace[-1,i]
        print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

    S = myEmcee.Emcee(pars+[logp],cov=cov/5.,nthreads=nthr,nwalkers=50)
    S.sample(300)
    outFile = '/data/ljo31b/EELs/esi/kinematics/inference/'+outfile
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()
    lp,trace,dic,_ = S.result()
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = trace[a1,a2,i]
        print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

    velL,sigL,velS,sigS = trace[a1,a2]
    
    zL, zS = zl+velL/light, zs+velS/light
    linwave = 10**outwave
    oper[:ntemps1] = t1.getSpectra(linwave,zL,sigL)
    oper[ntemps1:2.*ntemps1,~cond] = t1.getSpectra(linwave[~cond],zS,sigS)
    oper[2*ntemps1:2*ntemps1+ntemps2,cond] = t2.getSpectra(linwave[cond],zS,sigS)


    op = (oper*isig).T
    rhs = (scispec+BIAS)*isig
    fit,chi = optimize.nnls(op,rhs)
    for i in range(nfit):
        fit[ntemps1*2+ntemps2+i] -= bias#/grid**i
    outmodel = scipy.dot(oper.T,fit)
    continuum = scipy.dot(operator,fit)

    return S.result()

def readresults(scispec,varspec,t1,t2,outwave,zl,zs,nfit=12,infile=None,mask=None,bias=1e8,lim=5000.,bg='polynomial',restmask=None,srclim=6000.,maxlim=8000.):
    outwave,scispec,varspec = outwave[outwave>log10(lim)], scispec[outwave>log10(lim)], varspec[outwave>log10(lim)]
    origwave,origsci,origvar = outwave.copy(),scispec.copy(),varspec.copy()
    if mask is not None:
        ma = np.ones(outwave.size)
        for M in mask:
            cond = np.where((outwave>np.log10(M[0]))&(outwave<np.log10(M[1])))
            ma[cond]=0
        if restmask is not None:
            for M in restmask:
                cond = np.where((outwave>np.log10(M[0]*(1.+zl)))&(outwave<np.log10(M[1]*(1.+zs))))
                ma[cond]=0
                cond = np.where((outwave>np.log10(M[0]*(1.+zs)))&(outwave<np.log10(M[1]*(1.+zs))))
                ma[cond]=0
        ma=ma==1
        outwave,scispec,varspec=outwave[ma],scispec[ma],varspec[ma]
    isig = 1./varspec**0.5
    ntemps1,ntemps2 = t1.nspex, t2.nspex

   # Create the polynomial fit components
    BIAS = scispec*0.
    #grid = 10**outwave[-1] - 10**outwave[0]
    operator = scipy.zeros((scispec.size,2*ntemps1+ntemps2+nfit))
    X = np.arange(outwave.size)
    ff = 2.*X/X.max() - 1.
    for i in range(nfit):
        p = scipy.zeros((nfit,1))
        p[i] = 1.
        coeff = {'coeff':p,'type':bg}
        poly = sf.genfunc(ff,0.,coeff)
        operator[:,i+2*ntemps1+ntemps2] = poly
        BIAS += bias*poly#/grid**i

    oper = operator.T 
    cond = np.where(outwave<=np.log10(srclim),True,False)

    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+infile)
    lp,trace,dic,_=result
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    velL,sigL,velS,sigS = trace[a1,a2]

    zL, zS = zl+velL/light, zs+velS/light
    linwave = 10**outwave
    oper[:ntemps1] = t1.getSpectra(linwave,zL,sigL)
    oper[ntemps1:2.*ntemps1,~cond] = t1.getSpectra(linwave[~cond],zS,sigS)
    oper[2*ntemps1:2*ntemps1+ntemps2,cond] = t2.getSpectra(linwave[cond],zS,sigS)
    
    op = (oper*isig).T
    rhs = (scispec+BIAS)*isig
    fit,chi = optimize.nnls(op,rhs)
    for i in range(nfit):
        fit[ntemps1*2+ntemps2+i] -= bias#/grid**i
    maskmodel = scipy.dot(oper.T,fit)
    
    # unmasked
    if mask is not None or restmask is not None:
        operator = scipy.zeros((origsci.size,2*ntemps1+ntemps2+nfit))
        X = np.arange(origwave.size)
        ff = 2.*X/X.max() - 1.

        for i in range(nfit):
            p = scipy.zeros((nfit,1))
            p[i] = 1.
            coeff = {'coeff':p,'type':bg}
            poly = sf.genfunc(ff,0.,coeff)
            operator[:,i+2*ntemps1+ntemps2] = poly

        oper = operator.T 
        origcond = np.where(origwave<=np.log10(srclim),True,False)
        linorigwave = 10**origwave
                       
    oper[:ntemps1] = t1.getSpectra(linorigwave,zL,sigL)
    oper[ntemps1:2.*ntemps1,~origcond] = t1.getSpectra(linorigwave[~origcond],zS,sigS)
    oper[2*ntemps1:2*ntemps1+ntemps2,origcond] = t2.getSpectra(linorigwave[origcond],zS,sigS)
    
    outmodel = scipy.dot(oper.T,fit)
    lens = scipy.dot(oper[:ntemps1].T,fit[:ntemps1]) 
    source = scipy.dot(oper[2*ntemps1:2*ntemps1+ntemps2].T,fit[2*ntemps1:2*ntemps1+ntemps2]) + scipy.dot(oper[ntemps1:2*ntemps1].T,fit[ntemps1:2*ntemps1])
    cont = scipy.dot(oper[2*ntemps1+ntemps2:].T,fit[2*ntemps1+ntemps2:])
    
    pl.figure()
    pl.subplot(211)
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
    pl.plot(10**origwave,source,'Crimson')
    pl.plot(10**origwave,cont,'Navy')
    pl.legend(loc='upper right',frameon=False)
    pl.ylabel('flux')
    pl.axis([lim,maxlim,-0.5,4])
    pl.figtext(0.15,0.85,r'$\sigma_{s} = $'+'%.2f'%sigS+'; $\sigma_{l} = $'+'%.2f'%sigL)
    pl.subplot(212)
    if mask is not None:
        for M in mask:
            pl.axvspan(M[0], M[1], color='DarkGray')
    if restmask is not None:
        for M in restmask:
            pl.axvspan(M[0]*(1.+zl),M[1]*(1.+zl),color='DarkGray')
            pl.axvspan(M[0]*(1.+zs), M[1]*(1.+zs),color='DarkGray')
    pl.plot(10**origwave,(origsci-outmodel),'k')
    pl.xlabel('observed wavelength ($\AA$)')
    pl.ylabel('residuals')
    pl.axis([lim,maxlim,-1,1])
    print '%.2f'%velL, '&','%.2f'%velS,'&', '%.2f'%sigL,'&','%.2f'%sigS
    for i in fit:
        print i
    return result

