      program bah
      end

      subroutine sersicdeflections(x,y,amp,r,eta,q,m,dr,k,xout,yout,n)
!f2py double precision dimension(n),intent(in) :: x,y
!f2py double precision deminsion(n),intent(out) :: xout,yout
!f2py integer intent(hide),depend(x) :: n=len(x)
!f2py integer intent(in), optional :: m = 31
!f2py double precision intent(in), optional :: dr = 5.
!f2py integer intent(in), optional :: k = 3
      double precision x(n),y(n),xout(n),yout(n)
      double precision amp,r,eta,q
      double precision xintegrand(m),yintegrand(m),uvar(m)
      double precision u,u2
      double precision x2,y2,q2,u2q2
      double precision lim,delta,w2
      double precision w(m)
      integer iopt,m,k,nest,lwrk,ier,n0
      double precision xb,xe,s,fp,d
      double precision xy,r2,dr,dr2,sq2
      double precision serK,norm,ie,reff
      double precision res,abserr
      double precision alist(m),blist(m),rlist(m),elist(m)
      integer neval,iord,last
      external xfunc,yfunc
      common /c1/ sq2,dr2,r2,xy,x2,y2,serK,reff,ie

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
      d = (m-1)/dr
      sq2 = 1.-q2

      serK = 2.*eta-1./3+4./(405.*eta)+46/(25515.*eta**2)
      norm = amp
      serK = serK*-1.
      ie = 1./eta
      reff = r
      do i=1,n
        x2 = x(i)*x(i)
        y2 = y(i)*y(i)
        lim = dsqrt(x2+y2/q2)
c        lim = dsqrt(x2+y2/q2)
        xy = x2*y2
        r2 = x2+y2
        dr2 = y2-x2
        call dqagse(xfunc,0.0001,lim,0.,1e-3.,m,res,abserr,neval,ier,
     *             alist,blist,rlist,elist,iord,last)
        xout(i) = 2*norm*q*x(i)*res
        call dqagse(yfunc,0.0001,lim,0.,1e-3,m,res,abserr,neval,ier,
     *             alist,blist,rlist,elist,iord,last)
        yout(i) = 2*norm*q*y(i)*res
      enddo
      end

      function xfunc(u)
      double precision u,xfunc
      double precision r2,dr,dr2,sq2,x2,y2
      double precision u2,delta,w2,u2q2,serK,reff,ie
      common /c1/ sq2,dr2,r2,xy,x2,y2,serK,reff,ie

      u2 = u*u
      u2q2 = u2*sq2
      delta = dsqrt((u2q2+dr2)*(u2q2+dr2)+4.*xy)
      w2 = (delta+r2+u2q2)/(delta+r2-u2q2)
      xfunc = dexp(serK*(u/reff)**ie)*u*dsqrt(w2)/(x2+y2*w2*w2)
      return
      end

      function yfunc(u)
      double precision u,yfunc
      double precision r2,dr,dr2,sq2,x2,y2
      double precision u2,delta,w2,u2q2,serK,reff,ie
      common /c1/ sq2,dr2,r2,xy,x2,y2,serK,reff,ie

      u2 = u*u
      u2q2 = u2*sq2
      delta = dsqrt((u2q2+dr2)*(u2q2+dr2)+4.*xy)
      w2 = (delta+r2+u2q2)/(delta+r2-u2q2)
      yfunc = dexp(serK*(u/reff)**ie)*u*dsqrt(w2)*w2/(x2+y2*w2*w2)
      return
      end
