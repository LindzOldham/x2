      subroutine count(nrow,ncolb,ja,ia,jb,ib,iw,nnz)
!f2py integer intent(out) :: nnz
      integer ja(*),jb(*),ia(*),ib(*),iw(*),nnz
      integer nrow,ncolb

      do k=1, ncolb
         iw(k) = 0
      enddo

      nnz = 0

      do ii=1,nrow
         ldg = 0
         last = -1
         do j=ia(ii),ia(ii+1)-1
            jr = ja(j)
            do k=ib(jr),ib(jr+1)-1
               jjc = jb(k)
               if (iw(jjc) .eq. 0) then
                  ldg = ldg + 1
                  iw(jjc) = last
                  last = jjc
               end if
            enddo
         enddo 
         nnz = nnz+ldg

         do k=1,ldg
            j = iw(last)
            iw(last) = 0
            last = j
         enddo
      enddo
      end


      subroutine count_t(nrow,ja,ia,jb,ib,iw,nnz)
!f2py integer intent(out) :: nnz
      integer ja(*),ia(*),jb(*),ib(*),iw(*),nnz
      integer nrow

      do k=1, nrow
         iw(k) = 0
      enddo

      nnz = 0

      do ii=1,nrow
        ldg = 0
        last = -1
        do j=ia(ii),ia(ii+1)-1
          jr = ja(j)
          do k=ib(jr),ib(jr+1)-1
            jjc = jb(k)
            if (jjc.ge.ii) then
              exit
            endif
            if (iw(jjc).eq.0) then
              ldg = ldg + 1
              iw(jjc) = last
              last = jjc
            endif
          enddo
        enddo
        nnz = nnz+ldg

        do k=1,ldg
          j = iw(last)
          iw(last) = 0
          last = j
        enddo
      enddo
      nnz = nnz+nrow
      end
 

      subroutine multD(nrow,ncol,ja,ia,jb,ib,a,b,nnz,c,jc,ic)
!f2py integer depend(len(ia)),intent(out) :: ic(len(ia))
!f2py double precision dimension(nnz),intent(out) :: c
!f2py integer dimension(nnz),intent(out) :: jc
      integer ja(*),jb(*),ia(*),ib(*),nnz,jc(nnz),ic(*)
      integer next(ncol),head,temp,nrow,ncol
      double precision a(*),b(*),c(*),sums(ncol),v

      do i=1,ncol
        next(i) = -1
        sums(i) = 0.
      enddo

      nnz = 0
      ic(1) = 0

      do i=0,nrow-1
        head = -2
        length = 0

        jj_start = ia(i+1)
        jj_end = ia(i+2)
        do jj=jj_start,jj_end-1
          j = ja(jj+1)
          v = a(jj+1)

          kk_start = ib(j+1)
          kk_end = ib(j+2)
          do kk=kk_start,kk_end-1
            k = jb(kk+1)

            sums(k+1) = sums(k+1)+v*b(kk+1)
            if (next(k+1).eq.-1) then
              next(k+1) = head
              head = k
              length = length+1
            endif
          enddo
        enddo


        do jj=0,length-1
          if (sums(head+1).ne.0) then
            jc(nnz+1) = head
            c(nnz+1) = sums(head+1)
            nnz = nnz+1
          endif

          temp = head
          head = next(head+1)

          next(temp+1) = -1
          sums(temp+1) = 0
        enddo

        ic(i+2) = nnz
      enddo
      end


      subroutine sort_csr(n,ib,jb,b)
!f2py integer dimension(len(jb)),intent(inout) :: jb
!f2py double precision dimension(len(b)),intent(inout) :: b
      integer ib(*),jb(*)
      double precision b(*)

      do ii=1,n
        jj_s = ib(ii)
        jj_e = ib(ii+1)
        do i=2,jj_e-jj_s
          x1 = jb(i+jj_s)
          x2 = b(i+jj_s)
          j = i
10        j = j-1
          if (j.eq.0 .or. jb(j+jj_s).le.x1) go to 20
          jb(j+1+jj_s) = jb(j+jj_s)
          b(j+1+jj_s) = b(j+jj_s)
          go to 10
20        jb(j+1+jj_s) = x1
          b(j+1+jj_s) = x2
        enddo
      enddo
      end


      subroutine sort_csr3(n,m,ib,jb,b)
!f2py integer intent(hide),depend(jb) :: maxlen=len(ja)
!f2py integer dimension(len(jb)),intent(inout) :: jb
!f2py double precision dimension(len(b)),intent(inout) :: b
      integer ib(*),jb(*),jbC(m)
      double precision b(*),bC(m)
      integer clen,indx1,indx2
      integer mbit,bit

      mbit = 1073741824
10    mbit = RSHIFT(mbit,1)
      if (IAND(mbit,n).eq.0) go to 10
      mbit = LSHIFT(mbit,1)

      do ii=1,n
        jj_s = ib(ii)
        clen = ib(ii+1)-jj_s
        if (clen.lt.50) then
          do i=2,clen
            x1 = jb(i+jj_s)
            x2 = b(i+jj_s)
            j = i
20          j = j-1
            if (j.eq.0 .or. jb(j+jj_s).le.x1) go to 30
            jb(j+1+jj_s) = jb(j+jj_s)
            b(j+1+jj_s) = b(j+jj_s)
            go to 20
30          jb(j+1+jj_s) = x1
            b(j+1+jj_s) = x2
          enddo
        else
          bit = 1
40        indx1 = 1
          indx2 = 1
          do i=1,clen
              if(IAND(bit,jb(i+jj_s)).eq.0) indx1 = indx1+1
          enddo
          do i=1,clen
              if(IAND(bit,jb(i+jj_s)).eq.0) then
                  jbC(indx2) = jb(i+jj_s)
                  bC(indx2) = b(i+jj_s)
                  indx2 = indx2 + 1
              else
                jbC(indx1) = jb(i+jj_s)
                bC(indx1) = b(i+jj_s)
                indx1 = indx1+1
              endif
          enddo
          do i=1,clen
              jb(i+jj_s) = jbC(i)
              b(i+jj_s) = bC(i)
          enddo
          bit = LSHIFT(bit,1)
          if(bit.ne.mbit) go to 40
        endif
      enddo
      end


      subroutine sort_csr2(n,m,ib,jb,b)
!f2py integer dimension(len(jb)),intent(inout) :: jb
!f2py double precision dimension(len(b)),intent(inout) :: b
      integer ib(*),jb(*),wrk(n),tmpjb(m)
      double precision b(*),tmpb(m)

      do ii=1,n
        jj_s = ib(ii)+1
        jj_e = ib(ii+1)+1
c        call sortrx(jj_e-jj_s,jb(jj_s:jj_e),wrk(1:jj_e-jj_s))
        call qsorti(wrk(1:jj_e-jj_s),jj_e-jj_s,jb(jj_s:jj_e))
        do i=1,jj_e-jj_s
          tmpjb(i+jj_s-1) = jb(wrk(i)+jj_s-1)
          tmpb(i+jj_s-1) = b(wrk(i)+jj_s-1)
        enddo
      enddo

      do i=1,m
        b(i) = tmpb(i)
        jb(i) = tmpjb(i)
      enddo
      end


      subroutine multD_t(nrow,ncol,ja,ia,jb,ib,a,b,nnz,c,jc,ic)
!f2py integer depend(len(ia)),intent(out) :: ic(len(ia))
!f2py double precision dimension(nnz),intent(out) :: c
!f2py integer dimension(nnz),intent(out) :: jc
      integer ja(*),ia(*),ib(*),nnz,jc(nnz),ic(*)
      integer jb(*)
      integer next(ncol),head,temp,nrow,ncol
      double precision a(*),b(*),c(*),sums(ncol),v

      do i=1,ncol
        next(i) = -1
        sums(i) = 0.
      enddo

      nnz = 0
      ic(1) = 0

      do i=0,nrow-1
        head = -2
        length = 0

        jj_start = ia(i+1)
        jj_end = ia(i+2)
        do jj=jj_start,jj_end-1
          j = ja(jj+1)
          v = a(jj+1)

          kk_start = ib(j+1)
          kk_end = ib(j+2)
          do kk=kk_start,kk_end-1
            k = jb(kk+1)
            if (k.gt.i) then
c              print *,i,jj,kk-kk_start,kk_end-kk_start
              exit
            endif
            if (next(k+1).eq.-1) then
              next(k+1) = head
              head = k
              length = length+1
            endif
            sums(k+1) = sums(k+1)+v*b(kk+1)
          enddo
        enddo


        do jj=0,length-1
          if (sums(head+1).ne.0) then
            jc(nnz+1) = head
            if (head.eq.i) then
              c(nnz+1) = sums(head+1)/2
            else
              c(nnz+1) = sums(head+1)
            endif
            nnz = nnz+1
          endif

          temp = head
          head = next(head+1)

          next(temp+1) = -1
          sums(temp+1) = 0
        enddo

        ic(i+2) = nnz
      enddo
      end


      subroutine mult(nrow,ncol,ja,ia,jb,ib,a,b,nnz,c,jc,ic)
!f2py integer depend(len(ia)),intent(out) :: ic(len(ia))
!f2py double precision dimension(nnz),intent(out) :: c
!f2py integer dimension(nnz),intent(out) :: jc
      integer ja(*),jb(*),ia(*),ib(*),nnz,jc(nnz),ic(*)
      integer next(ncol),head,temp,nrow,ncol
      double precision a(*),b(*),c(nnz),sums(ncol),v

      do i=1,ncol
        next(i) = -1
        sums(i) = 0.
      enddo

      nnz = 1
      ic(1) = 1

      do i=1,nrow
        head = -2
        length = 0

        jj_start = ia(i)
        jj_end = ia(i+1)
        do jj=jj_start,jj_end
          j = ja(jj)
          v = a(jj)

          kk_start = ib(j)
          kk_end = ib(j+1)
          do kk=kk_start,kk_end
            k = jb(kk)

            sums(k) = sums(k)+v*b(kk)
            if(next(k).eq.-1) then
              next(k) = head
              head = k
              length = length+1
            endif
          enddo
        enddo

        do jj=1,length
          if(sums(head).ne.0) then
            jc(nnz) = head
            c(nnz) = sums(head)
            nnz = nnz+1
          endif
          temp = head
          head = next(head)

          next(temp) = -1
          sums(temp) = 0
        enddo
        ic(i+1) = nnz
      enddo
      end








      SUBROUTINE SORTRX(N,DATA,INDEX)
C===================================================================
C
C     SORTRX -- SORT, Real input, indeX output
C
C
C     Input:  N     INTEGER
C             DATA  REAL
C
C     Output: INDEX INTEGER (DIMENSION N)
C
C This routine performs an in-memory sort of the first N elements of
C array DATA, returning into array INDEX the indices of elements of
C DATA arranged in ascending order.  Thus,
C
C    DATA(INDEX(1)) will be the smallest number in array DATA;
C    DATA(INDEX(N)) will be the largest number in DATA.
C
C The original data is not physically rearranged.  The original order
C of equal input values is not necessarily preserved.
C
C===================================================================
C
C SORTRX uses a hybrid QuickSort algorithm, based on several
C suggestions in Knuth, Volume 3, Section 5.2.2.  In particular, the
C "pivot key" [my term] for dividing each subsequence is chosen to be
C the median of the first, last, and middle values of the subsequence;
C and the QuickSort is cut off when a subsequence has 9 or fewer
C elements, and a straight insertion sort of the entire array is done
C at the end.  The result is comparable to a pure insertion sort for
C very short arrays, and very fast for very large arrays (of order 12
C micro-sec/element on the 3081K for arrays of 10K elements).  It is
C also not subject to the poor performance of the pure QuickSort on
C partially ordered data.
C
C Created:  15 Jul 1986  Len Moss
C
C===================================================================
 
      INTEGER   N,INDEX(N)
      REAL      DATA(N)
 
      INTEGER   LSTK(31),RSTK(31),ISTK
      INTEGER   L,R,I,J,P,INDEXP,INDEXT
      REAL      DATAP
 
C     QuickSort Cutoff
C
C     Quit QuickSort-ing when a subsequence contains M or fewer
C     elements and finish off at end with straight insertion sort.
C     According to Knuth, V.3, the optimum value of M is around 9.
 
      INTEGER   M
      PARAMETER (M=11)
 
C===================================================================
C
C     Make initial guess for INDEX
 
      DO 50 I=1,N
         INDEX(I)=I
   50    CONTINUE
 
C     If array is short, skip QuickSort and go directly to
C     the straight insertion sort.
 
      IF (N.LE.M) GOTO 900
 
C===================================================================
C
C     QuickSort
C
C     The "Qn:"s correspond roughly to steps in Algorithm Q,
C     Knuth, V.3, PP.116-117, modified to select the median
C     of the first, last, and middle elements as the "pivot
C     key" (in Knuth's notation, "K").  Also modified to leave
C     data in place and produce an INDEX array.  To simplify
C     comments, let DATA[I]=DATA(INDEX(I)).
 
C Q1: Initialize
      ISTK=0
      L=1
      R=N
 
  200 CONTINUE
 
C Q2: Sort the subsequence DATA[L]..DATA[R].
C
C     At this point, DATA[l] <= DATA[m] <= DATA[r] for all l < L,
C     r > R, and L <= m <= R.  (First time through, there is no
C     DATA for l < L or r > R.)
 
      I=L
      J=R
 
C Q2.5: Select pivot key
C
C     Let the pivot, P, be the midpoint of this subsequence,
C     P=(L+R)/2; then rearrange INDEX(L), INDEX(P), and INDEX(R)
C     so the corresponding DATA values are in increasing order.
C     The pivot key, DATAP, is then DATA[P].
 
      P=(L+R)/2
      INDEXP=INDEX(P)
      DATAP=DATA(INDEXP)
 
      IF (DATA(INDEX(L)) .GT. DATAP) THEN
         INDEX(P)=INDEX(L)
         INDEX(L)=INDEXP
         INDEXP=INDEX(P)
         DATAP=DATA(INDEXP)
      ENDIF
 
      IF (DATAP .GT. DATA(INDEX(R))) THEN
         IF (DATA(INDEX(L)) .GT. DATA(INDEX(R))) THEN
            INDEX(P)=INDEX(L)
            INDEX(L)=INDEX(R)
         ELSE
            INDEX(P)=INDEX(R)
         ENDIF
         INDEX(R)=INDEXP
         INDEXP=INDEX(P)
         DATAP=DATA(INDEXP)
      ENDIF
 
C     Now we swap values between the right and left sides and/or
C     move DATAP until all smaller values are on the left and all
C     larger values are on the right.  Neither the left or right
C     side will be internally ordered yet; however, DATAP will be
C     in its final position.
 
  300 CONTINUE
 
C Q3: Search for datum on left >= DATAP
C
C     At this point, DATA[L] <= DATAP.  We can therefore start scanning
C     up from L, looking for a value >= DATAP (this scan is guaranteed
C     to terminate since we initially placed DATAP near the middle of
C     the subsequence).
 
         I=I+1
         IF (DATA(INDEX(I)).LT.DATAP) GOTO 300
 
  400 CONTINUE
 
C Q4: Search for datum on right <= DATAP
C
C     At this point, DATA[R] >= DATAP.  We can therefore start scanning
C     down from R, looking for a value <= DATAP (this scan is guaranteed
C     to terminate since we initially placed DATAP near the middle of
C     the subsequence).
 
         J=J-1
         IF (DATA(INDEX(J)).GT.DATAP) GOTO 400
 
C Q5: Have the two scans collided?
 
      IF (I.LT.J) THEN
 
C Q6: No, interchange DATA[I] <--> DATA[J] and continue
 
         INDEXT=INDEX(I)
         INDEX(I)=INDEX(J)
         INDEX(J)=INDEXT
         GOTO 300
      ELSE
 
C Q7: Yes, select next subsequence to sort
C
C     At this point, I >= J and DATA[l] <= DATA[I] == DATAP <= DATA[r],
C     for all L <= l < I and J < r <= R.  If both subsequences are
C     more than M elements long, push the longer one on the stack and
C     go back to QuickSort the shorter; if only one is more than M
C     elements long, go back and QuickSort it; otherwise, pop a
C     subsequence off the stack and QuickSort it.
 
         IF (R-J .GE. I-L .AND. I-L .GT. M) THEN
            ISTK=ISTK+1
            LSTK(ISTK)=J+1
            RSTK(ISTK)=R
            R=I-1
         ELSE IF (I-L .GT. R-J .AND. R-J .GT. M) THEN
            ISTK=ISTK+1
            LSTK(ISTK)=L
            RSTK(ISTK)=I-1
            L=J+1
         ELSE IF (R-J .GT. M) THEN
            L=J+1
         ELSE IF (I-L .GT. M) THEN
            R=I-1
         ELSE
C Q8: Pop the stack, or terminate QuickSort if empty
            IF (ISTK.LT.1) GOTO 900
            L=LSTK(ISTK)
            R=RSTK(ISTK)
            ISTK=ISTK-1
         ENDIF
         GOTO 200
      ENDIF
 
  900 CONTINUE
 
C===================================================================
C
C Q9: Straight Insertion sort
 
      DO 950 I=2,N
         IF (DATA(INDEX(I-1)) .GT. DATA(INDEX(I))) THEN
            INDEXP=INDEX(I)
            DATAP=DATA(INDEXP)
            P=I-1
  920       CONTINUE
               INDEX(P+1) = INDEX(P)
               P=P-1
               IF (P.GT.0) THEN
                  IF (DATA(INDEX(P)).GT.DATAP) GOTO 920
               ENDIF
            INDEX(P+1) = INDEXP
         ENDIF
  950    CONTINUE
 
C===================================================================
C
C     All done
 
      END






      SUBROUTINE QSORTI (ORD,N,A)
C
C==============SORTS THE ARRAY A(I),I=1,2,...,N BY PUTTING THE
C   ASCENDING ORDER VECTOR IN ORD.  THAT IS ASCENDING ORDERED A
C   IS A(ORD(I)),I=1,2,...,N; DESCENDING ORDER A IS A(ORD(N-I+1)),
C   I=1,2,...,N .  THIS SORT RUNS IN TIME PROPORTIONAL TO N LOG N .
C
C
C     ACM QUICKSORT - ALGORITHM #402 - IMPLEMENTED IN FORTRAN 66 BY
C                                 WILLIAM H. VERITY, WHV@PSUVM.PSU.EDU
C                                 CENTER FOR ACADEMIC COMPUTING
C                                 THE PENNSYLVANIA STATE UNIVERSITY
C                                 UNIVERSITY PARK, PA.  16802
C
      IMPLICIT INTEGER (A-Z)
C
      DIMENSION ORD(N),POPLST(2,20)
      INTEGER X,XX,Z,ZZ,Y
C
C     TO SORT DIFFERENT INPUT TYPES, CHANGE THE FOLLOWING
C     SPECIFICATION STATEMENTS; FOR EXAMPLE, FOR FORTRAN CHARACTER
C     USE THE FOLLOWING:  CHARACTER *(*) A(N)
C
      INTEGER A(N)
C
      NDEEP=0
      U1=N
      L1=1
      DO 1  I=1,N
    1 ORD(I)=I
    2 IF (U1.LE.L1) RETURN
C
    3 L=L1
      U=U1
C
C PART
C
    4 P=L
      Q=U
C     FOR CHARACTER SORTS, THE FOLLOWING 3 STATEMENTS WOULD BECOME
C     X = ORD(P)
C     Z = ORD(Q)
C     IF (A(X) .LE. A(Z)) GO TO 2
C
C     WHERE "CLE" IS A LOGICAL FUNCTION WHICH RETURNS "TRUE" IF THE
C     FIRST ARGUMENT IS LESS THAN OR EQUAL TO THE SECOND, BASED ON "LEN"
C     CHARACTERS.
C
      X=A(ORD(P))
      Z=A(ORD(Q))
      IF (X.LE.Z) GO TO 5
      Y=X
      X=Z
      Z=Y
      YP=ORD(P)
      ORD(P)=ORD(Q)
      ORD(Q)=YP
    5 IF (U-L.LE.1) GO TO 15
      XX=X
      IX=P
      ZZ=Z
      IZ=Q
C
C LEFT
C
    6 P=P+1
      IF (P.GE.Q) GO TO 7
      X=A(ORD(P))
      IF (X.GE.XX) GO TO 8
      GO TO 6
    7 P=Q-1
      GO TO 13
C
C RIGHT
C
    8 Q=Q-1
      IF (Q.LE.P) GO TO 9
      Z=A(ORD(Q))
      IF (Z.LE.ZZ) GO TO 10
      GO TO 8
    9 Q=P
      P=P-1
      Z=X
      X=A(ORD(P))
C
C DIST
C
   10 IF (X.LE.Z) GO TO 11
      Y=X
      X=Z
      Z=Y
      IP=ORD(P)
      ORD(P)=ORD(Q)
      ORD(Q)=IP
   11 IF (X.LE.XX) GO TO 12
      XX=X
      IX=P
   12 IF (Z.GE.ZZ) GO TO 6
      ZZ=Z
      IZ=Q
      GO TO 6
C
C OUT
C
   13 CONTINUE
      IF (.NOT.(P.NE.IX.AND.X.NE.XX)) GO TO 14
      IP=ORD(P)
      ORD(P)=ORD(IX)
      ORD(IX)=IP
   14 CONTINUE
      IF (.NOT.(Q.NE.IZ.AND.Z.NE.ZZ)) GO TO 15
      IQ=ORD(Q)
      ORD(Q)=ORD(IZ)
      ORD(IZ)=IQ
   15 CONTINUE
      IF (U-Q.LE.P-L) GO TO 16
      L1=L
      U1=P-1
      L=Q+1
      GO TO 17
   16 U1=U
      L1=Q+1
      U=P-1
   17 CONTINUE
      IF (U1.LE.L1) GO TO 18
C
C START RECURSIVE CALL
C
      NDEEP=NDEEP+1
      POPLST(1,NDEEP)=U
      POPLST(2,NDEEP)=L
      GO TO 3
   18 IF (U.GT.L) GO TO 4
C
C POP BACK UP IN THE RECURSION LIST
C
      IF (NDEEP.EQ.0) GO TO 2
      U=POPLST(1,NDEEP)
      L=POPLST(2,NDEEP)
      NDEEP=NDEEP-1
      GO TO 18
C
C END SORT
C END QSORT
C
      END

