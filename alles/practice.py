## element by element version
name,x,y,pa,q,re,n,dx,dy,dpa,dq,dre,dn,sigma,dsigma,mu,dmu,mag,dmag,rho = np.load('/data/ljo31b/MACSJ0717/models/FPParams.npy').T
re *= 0.065*6.482
dre *= 0.065*6.482

xx,yy,zz = np.log10(sigma), mu+7.5*np.log10(1.55), np.log10(re)
dxx,dyy,dzz = dsigma/sigma, dmu.copy(),dre/re
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = rho*dyy*dzz, rho*dzz*dyy
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0

a,b,alpha,mux,muy=1.,0.3,-10.,1.,1.
taux,tauy,tauxy,tauyx = 1.,1.,1.,1.
sigma=0.1

taux2,tauy2,tauxy2,tauyx2,sigma2,a2,b2 = taux**2.,tauy**2.,tauxy**2.,tauyx**2.,sigma**2.,a**2.,b**2.
X = zz - alpha - a*mux - b*muy
Y = xx - mux
Z = yy - muy
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

Mxx = Syy*Szz-Szy*Syz
Mxy = Szx*Syz-Syx*Szz
Mxz = Syx*Szy-Syy*Szx
Myx = Sxz*Szy-Sxy*Szz
Myy = Sxx*Szz-Szx*Sxz
Myz = Sxy*Szx-Sxx*Szy
Mzx = Sxy*Syz-Sxz*Syy
Mzy = Sxz*Syx-Sxx*Syz
Mzz = Sxx*Syy-Sxy*Syx
Delta = Mxx*X**2. + Mxy*X*Y + Mxz*X*Z + Myx*X*Y + Myy*Y**2. + Myz*Y*Z + Mzx*X*Z + Mzy*Z*Y + Mzz*Z**2.
resid = -0.5*Delta/Sigma - 0.5*np.log(Sigma)


## matrix version
name,x,y,pa,q,re,n,dx,dy,dpa,dq,dre,dn,sigma,dsigma,mu,dmu,mag,dmag,rho = np.load('/data/ljo31b/MACSJ0717/models/FPParams.npy').T
re *= 0.065*6.482
dre *= 0.065*6.482

xx,yy,zz = np.log10(sigma), mu+7.5*np.log10(1.55), np.log10(re)
dxx,dyy,dzz = dsigma/sigma, dmu.copy(),dre/re
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = rho*dyy*dzz, rho*dzz*dyy
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0

a,b,alpha,mux,muy=1.,0.3,-10.,1.,1.
taux,tauy,tauxy,tauyx = 1.,1.,1.,1.
sigma=0.1

taux2,tauy2,tauxy2,tauyx2,sigma2,a2,b2 = taux**2.,tauy**2.,tauxy**2.,tauyx**2.,sigma**2.,a**2.,b**2.
X = zz - alpha - a*mux - b*muy
Y = xx - mux
Z = yy - muy
Sxx = taux2*a2 + a*b*(tauxy2+tauyx2) + tauy2*b2 + sigma2 + szz
Syy = taux2 + sxx
Szz = tauy2 + syy
Sxy = a*taux2 + b*tauyx2 + sxz
Sxz = a*tauxy2 + b*tauy2 + syz
Syx = taux2*a + tauxy2*b + sxz
Syz = tauxy2 + sxy
Szy = tauyx2 + syx
Szx = tauxy2*a + tauy2*b + syz
resid = 0
args = np.zeros(xx.size)
for ii in range(xx.size):
    V = np.matrix([[Sxx[ii], Sxy, Sxz],[Syx,Syy[ii],Syz],[Szx,Szy,Szz[ii]]])
    Vinv = V.I
    Vdet = np.linalg.det(V)
    Z = np.matrix([[zz[ii],xx[ii],yy[ii]]]).T
    args[ii] = -0.5*np.dot(Z.T,np.dot(Vinv,Z))# - 0.5*np.log(Vdet)
    resid += args[ii]
print resid
