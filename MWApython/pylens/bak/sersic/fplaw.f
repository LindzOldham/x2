      program bah
      end

      subroutine plawdeflections(x,y,rein,eta,q,xout,yout,n)
!f2py real dimension(n),intent(in) :: x,y
!f2py real deminsion(n),intent(out) :: xout,yout
!f2py integer intent(hide),depend(x) :: n=len(x)
      real x(n),y(n),xout(n),yout(n)
      real xintegrand(51),yintegrand(51),uvar(51)
      real u,u2
      real rein,eta,s,q,x2,y2,q2,u2q2
      real lim,delta,w2

      q2 = q*q
      do i=1,n
        x2 = x(i)*x(i)
        y2 = y(i)*y(i)
        lim = log10(x2+y2/q2)
        do j=1,51
          u = 10**(lim-5+(j-1)/10.)
          u2 = u*u
          u2q2 = u2*(1-q2)
          delta = ((u2q2+(y2-x2))**2+4*x2*y2)**0.5
          w2 = (delta+x2+y2+u2q2)/(delta+x2+y2-u2q2)
          xintegrand(j) = (2-eta)*u*w2**0.5/(x2+y2*(w2**2)*u**eta)
          yintegrand(j) = w2*xintegrand(j)
          uvar(j) = u
        enddo

        xout(i) = rein*x(i)*q**0.5*doint(uvar,xintegrand)
        yout(i) = rein*y(i)*q**0.5*doint(uvar,yintegrand)
      enddo
      end

      real function doint(u,integrand)
      real u(51),integrand(51),w(51)
      real xb,xe,s,fp
      integer iopt,m,k,nest,n,lwrk,ier
      real t(55),c(55),wrk(1084)
      integer iwrk(55)

      m = 51
      k = 3
      nest = 55
      lwrk = 1084
      iopt = 0
      xb = u(1)
      xe = u(51)
      s = 0.
      do i=1,51
        w(i) = 1.
      enddo
      call curfit(iopt,m,u,integrand,w,xb,xe,k,s,nest,n,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
      doint = splint(t,n,c,k,xb,xe,wrk)
      return
      end

      real function splint(t,n,c,k,a,b,wrk)
c  function splint calculates the integral of a spline function s(x)
c  of degree k, which is given in its normalized b-spline representation
c
c  calling sequence:
c     aint = splint(t,n,c,k,a,b,wrk)
c
c  input parameters:
c    t    : array,length n,which contains the position of the knots
c           of s(x).
c    n    : integer, giving the total number of knots of s(x).
c    c    : array,length n, containing the b-spline coefficients.
c    k    : integer, giving the degree of s(x).
c    a,b  : real values, containing the end points of the integration
c           interval. s(x) is considered to be identically zero outside
c           the interval (t(k+1),t(n-k)).
c
c  output parameter:
c    aint : real, containing the integral of s(x) between a and b.
c    wrk  : real array, length n.  used as working space
c           on output, wrk will contain the integrals of the normalized
c           b-splines defined on the set of knots.
c
c  other subroutines required: fpintb.
c
c  references :
c    gaffney p.w. : the calculation of indefinite integrals of b-splines
c                   j. inst. maths applics 17 (1976) 37-41.
c    dierckx p. : curve and surface fitting with splines, monographs on
c                 numerical analysis, oxford university press, 1993.
c
c  author :
c    p.dierckx
c    dept. computer science, k.u.leuven
c    celestijnenlaan 200a, b-3001 heverlee, belgium.
c    e-mail : Paul.Dierckx@cs.kuleuven.ac.be
c
c  latest update : march 1987
c
c  ..scalar arguments..
      real a,b
      integer n,k
c  ..array arguments..
      real t(n),c(n),wrk(n)
c  ..local scalars..
      integer i,nk1
c  ..
      nk1 = n-k-1
c  calculate the integrals wrk(i) of the normalized b-splines
c  ni,k+1(x), i=1,2,...nk1.
      call fpintb(t,n,wrk,nk1,a,b)
c  calculate the integral of s(x).
      splint = 0.
      do 10 i=1,nk1
        splint = splint+c(i)*wrk(i)
  10  continue
      return
      end
