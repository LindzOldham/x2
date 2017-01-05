import numpy as np, pylab as pl, pyfits as py
import cPickle
from tools import tools
from scipy.interpolate import splrep, splev, splint, spalde
from scipy import integrate
from scipy.special import gamma
from linslens import PROFILES

''' Behroozi 2010'''
def buildhalo(Mstar):
    ''' z=0 '''
    M1, beta, Mstar0, delta, gamma = 12.35, 0.44, 10.72, 0.57, 1.56
    mM = Mstar/10**Mstar0
    frac = mM**delta / (1.+mM**(-gamma))
    return M1 + beta*np.log10(mM) + frac - 0.5


def highz_halo(Mstar,a): # need to also have a(z)
    ''' 0 < z < 1 '''
    M1, beta, Mstar0, delta, gamma = 12.35, 0.44, 10.72, 0.57, 1.56
    M1a, betaa, Mstar0a, deltaa, gammaa = 0.28, 0.18, 0.55, 0.17, 2.51
    ''' evolve with scale factor a '''
    bta += betaa*(a-1.)
    delta += deltaa*(a-1.)
    gamma += gammaa*(a-1.)
    M1 += M1a*(a-1.)
    Mstar0 += Mstar0a*(a-1.)
    ''' then do the same as above '''
    mM = Mstar/10**Mstar0
    frac = mM**delta / (1.+mM**(-gamma))
    return M1 + beta*np.log10(mM) + frac - 0.5

def highz_halo_table(Mstar,z):
    logMh = np.arange(11.,15.2,0.25)
    if z == 0.5:
        logMh = logMh[1:-2]
        logMsh = np.array([-2.11,-1.84,-1.70,-1.68,-1.72,-1.81,-1.92,-2.05,-2.19,-2.34,-2.51,-2.68,-2.86,-3.04])
    elif z == 1.0:
        logMh = logMh[2:-4]
        logMsh = np.array([-2.01,-1.85,-1.77,-1.77,-1.81,-1.89,-1.99,-2.11,-2.25,-2.39,-2.55])
    else: 
        return
    logMs = logMsh+logMh
    model = splrep(logMs,logMh)
    return splev(Mstar,model)

def virialRadius(Mh,z):
    rhoc = tools.criticalDensity(z)
    return tools.virialRadius(Mh,rhoc,200.)

def scale_factor(z):
    ''' Komatsu 2009 cosmology '''
    om,h,s8 = 0.27,0.7,0.8
    ol=1.-om
    return a

def deproject(r,sb):
    ''' deproject a 2D profile assuming spherical symmetry '''
    sbmodel = splrep(r,sb)
    d_sb = np.array(spalde(r,sbmodel))[:,1]
    dmodel = splrep(r,d_sb)
    lr=r[:-100]
    light = lr*0.
    for i in range(lr.size):
        R = lr[i]
        rr = np.logspace(-5,0.5*np.log10(r[-1]**2.-R**2),1001)
        y = (rr**2. + R**2.)**0.5
        f = splev(y,dmodel)
        model = splrep(rr, -1.*f/(rr**2 + R**2)**0.5/np.pi)
        light[i] = splint(rr[0],rr[-1],model)
    return lr,light

def veldisp(r,sb,M,ap=None):
    ''' ap=aperture radius integrates over a circular aperture '''
    G=4.3e-6
    lr,light = deproject(r,sb)
    lM = light*M
    sr = lr[:-200]
    Isigma2 = sr*0.
    for i in range(sr.size):
        r=lr[i:]
        R=lr[i]
        model = splrep(r,lM[i:]*(r**2-R**2)**0.5/r**2)
        Isigma2[i] = 2*G*splint(R,lr[-1],model)
    
    if type(ap) in [float, np.float64]:
        radsigma = splrep(sr,sr*Isigma2)
        radsbmodel = splrep(sr,sr*sb[:-300])
        sigma2 = splint(0,ap,radsigma)/splint(0,ap,radsbmodel)
        return sigma2
    elif len(ap)>4.:
        radsigma = splrep(sr,sr*Isigma2)
        radsbmodel = splrep(sr,sr*sb[:-300])
        sigma2 = [splint(0,ap[i],radsigma)/splint(0,ap[i],radsbmodel) for i in range(ap.size)]
        sigma2 = np.array(sigma2)
        return sigma2    
    elif len(ap)==4.:
        x1,x2,y1,y2 = ap
        
        radsigma = splrep(sr,Isigma2)
        radsbmodel = splrep(sr,sb[:-300])
        
        sigmaintegrand = lambda x,y: splev((x**2+y**2)**0.5,radsigma)
        sbintegrand = lambda x,y: splev((x**2+y**2)**0.5,radsbmodel)
        
        integral2 = integrate.dblquad(sigmaintegrand,x1,x2,lambda y:y1,lambda y:y2)[0]/integrate.dblquad(sbintegrand,x1,x2,lambda y:y1,lambda y:y2)[0]
        return integral2
    sigma2 = Isigma2/sb[:-300]
    return sr,Isigma2#sigma2

def gridveldisp(arr):
    ''' ap=aperture radius integrates over a circular aperture '''
    r,sb,M,ap=arr
    G=4.3e-6
    lr,light = deproject(r,sb)
    lM = light*M
    sr = lr[:-200]
    Isigma2 = sr*0.
    for i in range(sr.size):
        r=lr[i:]
        R=lr[i]
        model = splrep(r,lM[i:]*(r**2-R**2)**0.5/r**2)
        Isigma2[i] = 2*G*splint(R,lr[-1],model)
    if type(ap) in [float, np.float64]:
        radsigma = splrep(sr,sr*Isigma2)
        radsbmodel = splrep(sr,sr*sb[:-300])
        sigma2 = splint(0,ap,radsigma)/splint(0,ap,radsbmodel)
        return sigma2
    elif len(ap)>4.:
        radsigma = splrep(sr,sr*Isigma2)
        radsbmodel = splrep(sr,sr*sb[:-300])
        sigma2 = [splint(0,ap[i],radsigma)/splint(0,ap[i],radsbmodel) for i in range(ap.size)]
        sigma2 = np.array(sigma2)
        return sigma2     
    elif len(ap)==4.:
        x1,x2,y1,y2 = ap
        
        radsigma = splrep(sr,Isigma2)
        radsbmodel = splrep(sr,sb[:-300])
        
        sigmaintegrand = lambda x,y: splev((x**2+y**2)**0.5,radsigma)
        sbintegrand = lambda x,y: splev((x**2+y**2)**0.5,radsbmodel)
        
        integral2 = integrate.dblquad(sigmaintegrand,x1,x2,lambda y:y1,lambda y:y2)[0]/integrate.dblquad(sbintegrand,x1,x2,lambda y:y1,lambda y:y2)[0]
        return integral2

    sigma2 = Isigma2/sb[:-300]
    return sr,Isigma2#sigma2


def veldisp_cbeta(r,sb,M,beta,ap=None):
    from scipy.special import hyp2f1 as hyp
    from scipy.special import betainc
    from scipy.special import gamma as GG

    Beta = lambda a,b,w: (w**a)/a * hyp(a,1.-b,a+1,w)
    G=4.3e-6
    lr,light = deproject(r,sb)
    lM = light*M
    model = splrep(lr,lM)
    sr = lr[:-200]
    Isigma2 = sr*0.
    
    t1 = (1.5-beta)*(np.pi**0.5)*GG(beta-0.5)/GG(beta)
    p1 = beta*GG(beta+0.5)*GG(0.5)/GG(1.+beta)
    p2 = GG(beta-0.5)*GG(0.5)/GG(beta)

    if beta>0.5:
        for i in range(sr.size):
            reval = np.logspace(np.log10(sr[i]),np.log10(sr[-1]),301)
            reval[0] = sr[i] # Avoid sqrt(-epsilon)
            Mlight = splev(reval,model)
            u = reval/sr[i]
            K = 0.5*(u**(2*beta-1.))*(t1+p1*betainc(beta+0.5,0.5,1./u**2)-p2*betainc(beta-0.5,0.5,1/u**2))
            mod = splrep(reval,K*Mlight/reval,k=3,s=0)
            Isigma2[i] = 2.*G*splint(sr[i],reval[-1],mod)
    elif beta<0.5:
        for i in range(sr.size):
            reval = np.logspace(np.log10(sr[i]),np.log10(sr[-1]),301)
            reval[0] = sr[i] # Avoid sqrt(-epsilon)
            Mlight = splev(reval,model)
            u = reval/sr[i]
            K = 0.5*(u**(2*beta-1.))*(t1+beta*Beta(beta+0.5,0.5,1./u**2)-Beta(beta-0.5,0.5,1/u**2))
            mod = splrep(reval,K*Mlight/reval,k=3,s=0)
            Isigma2[i] = 2.*G*splint(sr[i],reval[-1],mod)
    Isigma2[-1] = 0.
    if ap is not None:
        radsigma = splrep(sr,sr*Isigma2)
        radsbmodel = splrep(sr,sr*sb[:-300])
        if type(ap) in [float, np.float64]:
            sigma2 = splint(0,ap,radsigma)/splint(0,ap,radsbmodel)
        else:
            sigma2 = [splint(0,ap[i],radsigma)/splint(0,ap[i],radsbmodel) for i in range(ap.size)]
            sigma2 = np.array(sigma2)
        return sigma2     
    sigma2 = Isigma2/sb[:-300]
    return sr,sigma2

def NFW(r,rvir,Mvir,z=0.):
    ''' assuming Mvir is M200 and using the Maccio 2008 mass-concentration relation for WMAP3 cosmology '''
    om,ol,h = 0.3,0.7,0.7
    c = 0.830 - np.log10(ol+om*(1.+z)**3)/3. - 0.098*np.log10(h/1e12)
    c = 10**(c - 0.098*Mvir)
    r0 = rvir/c
    norm = 4.*np.pi*r0**3. * (np.log(1.+c) - (c/(1.+c)))
    rho0 = 10**Mvir/norm
    rr = r/r0
    rho = rho0/rr/(1+rr)**2.
    M = rho0*4.*np.pi*r0**3. * (np.log(1+(r/r0)) - (1./(1.+(r0/r))))
    return  M

def gNFW(r,r0,rvir,Mvir,gamma):
    ''' NB. this is an M87-like gNFW, not a lensing one!!! 
     specifying scale radius '''
    omega = 3.-gamma
    from scipy.special import hyp2f1 as hyp
    norm = 4.*np.pi*(r0**3.) * ((rvir/r0)**omega) * hyp(omega,omega,1+omega,-(rvir/r0)) / omega
    rho0 = 10**Mvir / norm
    M = rho0 * 4.*np.pi*(r0**3.) * ((r/r0)**omega) * hyp(omega,omega,1+omega,-(r/r0)) / omega
    return M


def light2mass(lr,light,ml):
    model = splrep(lr,ml*light*4.*np.pi*lr**2.)
    M = [splint(0,lr[i],model) for i in range(lr.size)]
    return np.array(M)


def gridgNFW(arr):
    lr,r0,gamma,zl,zs=arr
    mass = PROFILES.gNFW_TC(x=0.,y=0.,eta=gamma,rs=r0,pa=0.,q=1.,b=1.,zl=zl,zs=zs) 
    # renormalise as b changes
    M = mass.mass(lr)
    return M
    
def gridgNFW_sigma(arr):
    lr,r0,gamma,zl,zs,Rein_tot = arr
    mass = PROFILES.gNFW_TC(x=0.,y=0.,eta=gamma,rs=r0,pa=0.,q=1.,b=1.,zl=zl,zs=zs) 
    sigma = mass.sigma(lr)
    c = np.where(np.isfinite(sigma)==True)
    sigmod = splrep(lr[c],sigma[c]*2.*np.pi*lr[c])
    M_ein = splint(0,Rein_tot,sigmod)
    #print gamma, zs,  M_ein
    return M_ein

def grid_PL(arr):
    lr,eta,zl,zs = arr
    mass = PROFILES.PowerLaw(x=0.,y=0.,eta=eta,pa=0.,q=1.,b=1.,zl=zl,zs=zs) # assume b=1, then multiply sigma^2 by b**eta
    M = mass.mass(lr)
    return M

def gridPL_sigma(arr):
    lr,eta,zl,zs,Rein_tot = arr
    # this is being measured within TOTAL R_ein, NOT DARK R_ein!!!
    mass = PROFILES.PowerLaw(x=0.,y=0.,eta=eta,pa=0.,q=1.,b=1.,zl=zl,zs=zs) # assume b=1, then multiply sigma^2 by b**(2-eta)
    sigma = mass.sigma(lr)
    sigmod = splrep(lr,sigma*2.*np.pi*lr)
    M_ein = splint(0,Rein_tot,sigmod)
    print M_ein
    return M_ein
    
'''def gridgNFW(arr):
    r0,rvir,Mvir,gamma = arr
    r = np.logspace(-5,5,1501)
    r = r[:-100]
    # specifying scale radius
    omega = 3.-gamma
    from scipy.special import hyp2f1 as hyp
    norm = 4.*np.pi*(r0**3.) * ((rvir/r0)**omega) * hyp(omega,omega,1+omega,-(rvir/r0)) / omega
    rho0 = 10**Mvir / norm
    M = rho0 * 4.*np.pi*(r0**3.) * ((r/r0)**omega) * hyp(omega,omega,1+omega,-(r/r0)) / omega
    return M

def gNFW2(r,rvir,Mvir,gamma,z=0.):
    # assuming Mvir is M200 and using the Maccio 2008 mass-concentration relation for WMAP3 cosmology 
    om,ol,h = 0.3,0.7,0.7
    c = 0.830 - np.log10(ol+om*(1.+z)**3)/3. - 0.098*np.log10(h/1e12)
    c = 10**(c - 0.098*Mvir)
    r0 = rvir/c
    print r0,c
    # to get mass:
    from scipy.special import hyp2f1 as hyp
    omega = 3.-gamma    
    norm = 4.*np.pi*(r0**3.) * (c**omega) * hyp(omega,omega,1+omega,-c) / omega
    rho0 = 10**Mvir / norm
    M = rho0 * 4.*np.pi*r0**3. * ((r/r0)**omega) * hyp(omega,omega,1+omega,-(r/r0)) / omega
    return M

def gridgNFW2(arr):
    r,rvir,Mvir,gamma,z = arr
    # assuming Mvir is M200 and using the Maccio 2008 mass-concentration relation for WMAP3 cosmology 
    om,ol,h = 0.3,0.7,0.7
    c = 0.830 - np.log10(ol+om*(1.+z)**3)/3. - 0.098*np.log10(h/1e12)
    c = 10**(c - 0.098*Mvir)
    r0 = rvir/c
    # to get mass:
    from scipy.special import hyp2f1 as hyp
    omega = 3.-gamma    
    norm = 4.*np.pi*(r0**3.) * (c**omega) * hyp(omega,omega,1+omega,-c) / omega
    rho0 = 10**Mvir / norm
    M = rho0 * 4.*np.pi*r0**3. * ((r/r0)**omega) * hyp(omega,omega,1+omega,-(r/r0)) / omega
    return M'''

#def gridgNFW3(arr):
#    r,r0,gamma = arr
#    ''' assuming Mvir is M200 and using the Maccio 2008 mass-concentration relation for WMAP3 cosmology -- converting from r0 to rvir this time!'''
#    # to get mass:
#    from scipy.special import hyp2f1 as hyp
#    omega = 3.-gamma    
#    M =  4.*np.pi*r0**3. * ((r/r0)**omega) * hyp(omega,omega,1+omega,-(r/r0)) / omega
#    return M


'''def gridgNFW3_sigma(arr):
    lr,r0,gamma,Rein = arr

    # to get density:
    rho = (lr/r0)**(-gamma) * (1. + (lr/r0))**(gamma-3.)
    light_model = splrep(lr,rho)
    sr = lr[:-100]
    Sigma = sr*0.
    
    for i in range(sr.size):
        R = lr[i]
        rr = np.logspace(-5,0.5*np.log10(lr[-1]**2-R**2),1001)
        y = (rr**2+R**2)**0.5
        f = splev(y,light_model)
        model = splrep(rr,f)
        Sigma[i] = 2.*splint(rr[0],rr[-1],model)

    model = splrep(sr,Sigma*2.*np.pi*sr)
    Sig_einstein = splint(0,Rein,model)
    return Sig_einstein'''
    
