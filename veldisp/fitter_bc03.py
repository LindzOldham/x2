import scipy,numpy,cPickle
import special_functions as sf
from scipy import ndimage,optimize,signal,interpolate
from numpy import linalg
from math import sqrt,log,log10
import pymc, myEmcee_blobs as myEmcee
import numpy as np, pylab as pl

light = 299792.458

def finddispersion(scispec,varspec,t,tmpwave,outwave,zl,zs,nfit=6,outfile=None,mask=None,bias=1e8,lim=6000.,bg='polynomial'):
    outwave,scispec,varspec = outwave[outwave>log10(lim)], scispec[outwave>log10(lim)], varspec[outwave>log10(lim)]

    if mask is not None:
        ma = np.ones(outwave.size)
        for M in mask:
            cond = np.where((outwave>np.log10(M[0]))&(outwave<np.log10(M[1])))
            ma[cond]=0
        mask=ma==1
        outwave,scispec,varspec = outwave[mask],scispec[mask],varspec[mask]
    isig = 1./varspec**0.5
    ntemps = len(t)
    print ntemps

    vL = pymc.Uniform('lens velocity',-550.,550.,value=0.)
    sL = pymc.Uniform('lens dispersion',5.,501.)
    vS = pymc.Uniform('source velocity',-550.,550.,value=0.)
    sS = pymc.Uniform('source dispersion',5.,501.)
    pars = [vL,sL,vS,sS]
    cov = np.array([10.,10.,10.,10.])

    # Create the polynomial fit components
    BIAS = scispec*0.
    grid = 10**outwave[-1] - 10**outwave[0]
    operator = scipy.zeros((scispec.size,2*ntemps+nfit))
    for i in range(nfit):
        p = scipy.zeros((nfit,1))
        p[i] = 1.
        coeff = {'coeff':p,'type':bg}
        poly = sf.genfunc(10**outwave,0.,coeff)
        operator[:,i+2*ntemps] = poly
        BIAS += bias*poly/grid**i
    oper = operator.T  

    @pymc.deterministic
    def logprob(value=0.,pars=pars):
        zL, zS = np.log10(1.+zl+vL.value/light), np.log10(1.+zs+vS.value/light)
        for k in range(ntemps):
            oper[k] = interpolate.bisplev(sL.value,outwave-zL,t[k])
            oper[k+ntemps] = interpolate.bisplev(sS.value,outwave-zS,t[k])
        op = (oper*isig).T
        rhs = (scispec+BIAS)*isig
        fit,chi = optimize.nnls(op,rhs)
        lp = -0.5*chi**2.
        return lp
    
    @pymc.observed
    def logp(value=0.,lp=logprob):
        return lp

    S = myEmcee.Emcee(pars+[logp],cov=cov,nthreads=8,nwalkers=60)
    S.sample(400)
    outFile = '/data/ljo31b/EELs/esi/kinematics/trials/'+outfile
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()
    lp,trace,dic,_ = S.result()
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    for i in range(len(pars)):
        pars[i].value = np.median(trace[200:,:,i])
        print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

    velL,sigL,velS,sigS = trace[a1,a2]
    zL, zS = np.log10(1.+zl+velL/light), np.log10(1.+zs+velS/light)

    for k in range(ntemps):
        oper[k] = interpolate.bisplev(sigL,outwave-zL,t[k])
        oper[k+ntemps] = interpolate.bisplev(sigS,outwave-zS,t[k])

    op = (oper*isig).T
    rhs = (scispec+BIAS)*isig
    fit,chi = optimize.nnls(op,rhs)
    for i in range(nfit):
        fit[ntemps*2+i] -= bias/grid**i
    outmodel = scipy.dot(oper.T,fit)
    continuum = scipy.dot(operator,fit)

    '''pl.figure()
    pl.plot(10**outwave,scispec)
    pl.plot(10**outwave,outmodel)
    pl.plot(10**outwave,continuum)
    pl.figure()
    pl.plot(lp)
    pl.show()'''
    return S.result()

def readresults(scispec,varspec,t,tmpwave,outwave,zl,zs,nfit=6,infile=None,mask=None,bias=1e8,lim=6000.,bg='polynomial'):
    outwave,scispec,varspec = outwave[outwave>log10(lim)], scispec[outwave>log10(lim)], varspec[outwave>log10(lim)]
    origwave,origsci,origvar = outwave.copy(),scispec.copy(),varspec.copy()

    if mask is not None:
        ma = np.ones(outwave.size)
        for M in mask:
            cond = np.where((outwave>np.log10(M[0]))&(outwave<np.log10(M[1])))
            ma[cond]=0
        ma=ma==1
        outwave,scispec,varspec = outwave[ma],scispec[ma],varspec[ma]
    isig = 1./varspec**0.5
    ntemps = len(t)
    
    # Create the polynomial fit components
    BIAS = scispec*0.
    grid = 10**outwave[-1] - 10**outwave[0]
    operator = scipy.zeros((scispec.size,2*ntemps+nfit))
    for i in range(nfit):
        p = scipy.zeros((nfit,1))
        p[i] = 1.
        coeff = {'coeff':p,'type':bg}
        poly = sf.genfunc(10**outwave,0.,coeff)
        operator[:,i+2*ntemps] = poly
        BIAS += bias*poly/grid**i

    oper = operator.T 

    result = np.load('/data/ljo31b/EELs/esi/kinematics/trials/'+infile)
    lp,trace,dic,_=result
    a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
    velL,sigL,velS,sigS = np.median(trace[200:,:].reshape(((trace.shape[0]-200)*trace.shape[1],trace.shape[2])),axis=0)
    print velL, sigL, velS, sigS

    zL, zS = np.log10(1.+zl+velL/light), np.log10(1.+zs+velS/light)
    for k in range(ntemps):
        oper[k] = interpolate.bisplev(sigL,outwave-zL,t[k])
        oper[k+ntemps] = interpolate.bisplev(sigS,outwave-zS,t[k])

    op = (oper*isig).T
    rhs = (scispec+BIAS)*isig
    fit,chi = optimize.nnls(op,rhs)
    for i in range(nfit):
        fit[ntemps*2+i] -= bias/grid**i
    maskmodel = scipy.dot(oper.T,fit)
    # unmasked
    if mask is not None:
        operator = scipy.zeros((origsci.size,2*ntemps+nfit))
        grid = 10**origwave[-1] - 10**origwave[0]
        for i in range(nfit):
            p = scipy.zeros((nfit,1))
            p[i] = 1.
            coeff = {'coeff':p,'type':bg}
            poly = sf.genfunc(10**origwave,0.,coeff)
            operator[:,i+2*ntemps] = poly

        oper = operator.T 

        for k in range(ntemps):
            oper[k] = interpolate.bisplev(sigL,origwave-zL,t[k])
            oper[k+ntemps] = interpolate.bisplev(sigS,origwave-zS,t[k])
           

    outmodel = scipy.dot(oper.T,fit)
    lens = scipy.dot(oper[:ntemps].T,fit[:ntemps])
    source = scipy.dot(oper[ntemps:2*ntemps].T,fit[ntemps:2*ntemps])
    cont = scipy.dot(oper[2*ntemps:].T,fit[2*ntemps:])

    print oper[ntemps*2:].shape, oper[ntemps:ntemps*2].shape, ntemps, ntemps*2

    pl.figure()
    pl.subplot(211)
    if mask is not None:
        for M in mask:
            pl.axvspan(M[0], M[1], color='DarkGray')
    pl.plot(10**origwave,origsci,'LightGray',)
    pl.plot(10**origwave,outmodel,'k',)
    pl.plot(10**origwave,lens,'SteelBlue')
    pl.plot(10**origwave,source,'Crimson')
    pl.plot(10**origwave,cont,'Navy')
    pl.legend(loc='upper right',frameon=False)
    pl.ylabel('flux')
    pl.axis([lim,9000,-0.5,4])
    pl.figtext(0.15,0.85,r'$\sigma_{s} = $'+'%.2f'%sigS+'; $\sigma_{l} = $'+'%.2f'%sigL)
    pl.subplot(212)
    if mask is not None:
        for M in mask:
            pl.axvspan(M[0], M[1], color='DarkGray')
    # smooth residual spectrum
    resid = (origsci-outmodel)/origvar
    #ii = np.mod(resid.size,5)
    #resid = np.sort(resid[:-ii].reshape((int(resid.size)/5,5)),1).sum(1)
    ow = origwave#[:-ii][::5]
    pl.plot(10**ow,resid,'k')
    pl.xlabel('observed wavelength ($\AA$)')
    pl.ylabel('residuals')
    pl.axis([lim,9000,-120,120])

    print '%.2f'%velL, '&','%.2f'%velS, '&', '%.2f'%sigL, '&','%.2f'%sigS

    for i in fit:
        print i
    return result

