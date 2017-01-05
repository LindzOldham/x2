      program bah
      end

      subroutine sersicdeflections(x,y,amp,r,eta,q,m,k,xout,yout,n)
!f2py double precision dimension(n),intent(in) :: x,y
!f2py double precision deminsion(n),intent(out) :: xout,yout
!f2py integer intent(hide),depend(x) :: n=len(x)
!f2py integer intent(in), optional :: m = 31
!f2py integer intent(in), optional :: k = 3
      double precision x(n),y(n),xout(n),yout(n)
      double precision xintegrand(m),yintegrand(m),uvar(m)
      double precision u,u2
      double precision amp,eta,q,x2,y2,q2,u2q2
      double precision lim,delta,w2
      double precision w(m)
      integer iopt,m,k,nest,lwrk,ier,n0
      double precision t(m+k+1),c(m+k+1),wrk(m*(k+1)+(m+k+1)*(7+3*k))
      double precision xb,xe,s,fp,d
      integer iwrk(m+k+1)
      double precision r2,dr2,sq2
      double precision serK,norm,ie,r
      double precision ulo,uhi
      double precision kmu
      double precision fmu
      double precision pref,exparg
      double precision mu1,mu2,s1,s2,step

      nest = m+k+1
      lwrk = m*(k+1)+nest*(7+3*k)
      iopt = 0
      s = 0.
      do i=1,m
        w(i) = 1.
        uvar(i) = 0.
        xintegrand(i) = 0.
        yintegrand(i) = 0.
      enddo

      q5 = dsqrt(q)
      q2 = q*q
      d = (m-1)/5.
      sq2 = 1.-q2

      serK = 2.*eta-1./3+4./(405.*eta)+46/(25515.*eta**2)
      norm = amp*q/sq2
      serK = serK*-1.
      ie = 0.5/eta
      r2 = r*r
      do i=1,n
        s1 = sign(1.,x(i))
        s2 = sign(1.,y(i))
        x(i) = dabs(x(i))
        y(i) = dabs(y(i))
        x2 = x(i)*x(i)
        y2 = y(i)*y(i)
        ulo = 0.5*(y(i)/x(i)-x(i)/y(i))
        uhi = 0.5*(y(i)/(x(i)*q2)-x(i)*q2/y(i))
        step = (uhi-ulo)/(m-1.)
        pref = dsqrt(x(i)*y(i))
        do j=1,m
          u = ulo+step*(j-1.)
          u2 = u*u
          exparg = (2*u*x(i)*y(i)+x2-y2)
          if (dabs(exparg).lt.1e-10) exparg = 0.
          kmu =  dexp(serK*(exparg/(r2*sq2))**ie)
          mu1 = 1./dsqrt(1.+u2)
          mu2 = u/(u2+1.)
          xintegrand(j) = kmu*dsqrt(mu1-mu2)
          yintegrand(j) = kmu*dsqrt(mu1+mu2)
          uvar(j) = u
c          print *,u,kmu,dsqrt(mu1-mu2),exparg
        enddo
c        stop
        xb = uvar(1)
        xe = uvar(m)

        call curfit(iopt,m,uvar,xintegrand,w,xb,xe,k,s,nest,n0,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
        xout(i) = s1*norm*pref*splint(t,n0,c,k,xb,xe,wrk)
        call curfit(iopt,m,uvar,yintegrand,w,xb,xe,k,s,nest,n0,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
        yout(i) = s2*norm*pref*splint(t,n0,c,k,xb,xe,wrk)
      enddo
      end

