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



# Test of the full keyword
A_43 = big.(reshape(1:12, 4, 3))

U, S, V = svd(A_43, full=true)
@test U'*U ≈ Matrix(I, 4, 4)
@test V'*V ≈ Matrix(I, 3, 3)
@test U*[diagm(0 => S); zeros(1,3)]*V' ≈ A_43
@test issorted(S, rev=true)

U, S, V = svd(A_43, full=false)
@test U'*U ≈ Matrix(I, 3, 3)
@test V'*V ≈ Matrix(I, 3, 3)
@test U*diagm(0 => S)*V' ≈ A_43
@test issorted(S, rev=true)

A_34 = big.(reshape(1:12, 3, 4))
U, S, V = svd(A_34, full=true)
@test U'*U ≈ Matrix(I, 3, 3)
@test V'*V ≈ Matrix(I, 4, 4)
@test U*[diagm(0 => S) zeros(3,1)]*V' ≈ A_34
@test issorted(S, rev=true)

U, S, V = svd(A_34, full=false)
@test U'*U ≈ Matrix(I, 3, 3)
@test V'*V ≈ Matrix(I, 3, 3)
@test U*diagm(0 => S)*V' ≈ A_34
@test issorted(S, rev=true)
