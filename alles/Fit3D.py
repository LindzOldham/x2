import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle

pars, cov = [], []
pars.append(pymc.Uniform('a',-20,10,-10 ))
pars.append(pymc.Uniform('b',-20,10,-10 ))
pars.append(pymc.Uniform('alpha',-20,10,-10 ))
pars.append(pymc.Uniform('mu x',-20,10,-10 ))
pars.append(pymc.Uniform('mu y',-20,10,-10 ))
pars.append(pymc.Uniform('tau x',-20,10,-10 ))
pars.append(pymc.Uniform('tau y',-20,10,-10 ))
pars.append(pymc.Uniform('tau xy',-20,10,-10 ))
pars.append(pymc.Uniform('tau zx',-20,10,-10 )) # assume the same as above?!?
pars.append(pymc.Uniform('sigma',-20,10,-10 )) # assume the same as above?!?
cov += [1.,1.,1.,1.,1.,1.,1.,1.,1.,0.05]
optCov = np.array(cov)

# rewrite these as matrices!!!
@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    a,b,alpha,mux,muy,taux,tauy,tauxy,tauyx,sigma = pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value, pars[4].value, pars[5].value,pars[6].value, pars[7].value,pars[8].value, pars[9].value
    taux2,tauy2,tauxy2,tauyx2,sigma2,a2,b2 = taux**2.,tauy**2.,tauxy**2.,tauyx**2.,sigma**2.,a**2.,b**2.
    X = z - alpha - a*mux - b*muy
    Y = x - mux
    Z = y - muy
    Sxx = taux2*a2 + a*b*(tauxy2+tauyx2) + tauy2*b2 + sigma2 + szz
    Syy = taux2 + sxx
    Szz = tauy2 + syy
    Sxy = a*taux2 + b*tauyx2 + sxz
    Sxz = a*tauxy2 + b*tauy2 + syz
    Syx = taux2*a + tauxy2*b + sxz
    Syz = tauxy2 + sxy
    Szy = tauyx2 + syx
    Szx = tauxy2*a + tauy2*b + syz
    Sigma = Sxx*(Syy*Szz-Szy*Syz) - Sxy*(Syx*Szz-Szx*Syz) + Sxz*(Syx*Szy-Syy*Szx)
    ## make inverse matrix
    Mxx = Syy*Szz-Szy*Syz
    Mxy = Szx*Syz-Syx*Szz
    Mxz = Syx*Szy-Syy*Szx
    Myx = Sxz*Szy-Sxy*Szz
    Myy = Sxx*Szz-Szx*Sxz
    Myz = Sxy*Szx-Sxx*Szy
    Mzx = Sxy*Syz-Sxz*Syy
    Mzy = SxzSyx-Sxx*Syz
    Mzz = Sxx*Syy-Sxy*Syx
    Delta = Mxx*X**2. + Mxy*X*Y + Mxz*X*Z + Myx*X*Y + Myy*Y**2. + Myz*Y*Z + Mzx*X*Z + Mzy*Z*Y + Mzz*Z**2.
    resid = -0.5*Delta/Sigma - 0.5*np.log(Sigma)
    lp = resid.sum()
    return lp
    

X = x-mu
    Y = y - alpha - beta*mu
    Sxx = sxx + tau2
    Syy = syy + sigma2 + beta2*tau2
    Sxy = sxy + beta*tau2
    Syx = syx + beta*tau2
    Delta = Syy*X**2. + Sxx*Y**2. - X*Y*(Sxy+Syx)
    Sigma =  Sxx*Syy - Sxy*Syx
    pdf = -0.5*Delta/Sigma- 0.5*np.log(Sigma)
    lp = pdf.sum()
    return lp
