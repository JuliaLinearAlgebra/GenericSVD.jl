using GenericSVD
using Test, Random, LinearAlgebra

Random.seed!(1)

a = randexp()
b = randexp()
c = randn()
U = [a c; 0 b]
x,y = GenericSVD.svdvals2x2(a,b,c)

@test sort(sqrt.(eigvals(U'*U))) ≈ [x,y]
@test sort(svdvals(U)) ≈ [x,y]

U = Matrix(1.0I,3,3)
B = Bidiagonal([0.0,1.0,2.0],[3.0,4.0],:U)
B1 = copy(B)

GenericSVD.svd_zerodiag_row!(U,B,1,3)
@test B[1,1] == 0
@test B[1,2] == 0
@test U*Matrix(B) ≈ B1

Vt = Matrix(1.0I,3,3)
B = Bidiagonal([1.0,2.0,0.0],[3.0,4.0],:U)
B1 = copy(B)

GenericSVD.svd_zerodiag_col!(B,Vt,1,3)
@test B[3,3] == 0
@test B[2,3] == 0
@test Matrix(B)*Vt ≈ B1
