subroutine square_matrix(a,n,m)
    implicit none

    integer :: n,m
    real*8 :: a(n,m)
!f2py intent(in,out,copy) :: a
!f2py integer intent(hide),depend(a) :: n=shape(a,0), m=shape(a,1)
    a = matmul(a,a)

end