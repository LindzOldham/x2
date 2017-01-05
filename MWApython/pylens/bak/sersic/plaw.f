      program bah
      end

      subroutine plawdeflections(x,y,rein,eta,q,m,xout,yout,n)
!f2py double precision dimension(n),intent(in) :: x,y
!f2py double precision deminsion(n),intent(out) :: xout,yout
!f2py integer intent(hide),depend(x) :: n=len(x)
!f2py integer intent(in), optional :: m = 21
      double precision x(n),y(n),xout(n),yout(n)
      double precision xintegrand(m),yintegrand(m),uvar(m)
      double precision u,u2
      double precision rein,eta,q,x2,y2,q2,u2q2
      double precision lim,delta,w2,kappa
      double precision w(m)
      integer iopt,m,k,nest,n,lwrk,ier
      double precision t(m+4),c(m+4),wrk(m*4+(m+4)*16)
      double precision xb,xe,s,fp,d
      integer iwrk(m+4)
      double precision xy,r2,dr2,sq2

      k = 3
      nest = m+k+1
      lwrk = m*(k+1)+nest*(7+3*k)
      iopt = 0
      s = 0.
      do i=1,m
        w(i) = 1.
        uvar(i) = i*2.
        xintegrand(i) = 0.
        yintegrand(i) = 0.
      enddo

      q5 = dsqrt(q)
      q2 = q*q
      d = (m-1)/5.
      sq2 = 1.-q2
      rein = rein*q5*(2.-eta)
      do i=1,n
        x2 = x(i)*x(i)
        y2 = y(i)*y(i)
        lim = dlog10(dsqrt(x2+y2/q2))
c        lim = dsqrt(x2+y2/q2)
        xy = x2*y2
        r2 = x2+y2
        dr2 = y2-x2
        do j=1,m
          u = 10.**(lim-5+(j-1)/d)
c          u = lim*(j-0.9999)/m

          u2 = u*u
          u2q2 = u2*sq2
          delta = dsqrt((u2q2+dr2)*(u2q2+dr2)+4.*xy)
          w2 = (delta+r2+u2q2)/(delta+r2-u2q2)
          xintegrand(j) = u*dsqrt(w2)/(x2+y2*w2*w2)/u**eta
          yintegrand(j) = w2*xintegrand(j)
          uvar(j) = u
        enddo
        xb = uvar(1)
        xe = uvar(m)

        call curfit(iopt,m,uvar,xintegrand,w,xb,xe,k,s,nest,n,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
        xout(i) = rein*x(i)*splint(t,n,c,k,xb,xe,wrk)
        call curfit(iopt,m,uvar,yintegrand,w,xb,xe,k,s,nest,n,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
        yout(i) = rein*y(i)*splint(t,n,c,k,xb,xe,wrk)
      enddo
      end


