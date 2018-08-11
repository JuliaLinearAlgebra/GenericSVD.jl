using GenericSVD
using Test, Random, LinearAlgebra

Random.seed!(1)

n,m = 100,20

X = randn(n,m)
bX = big.(X)
bS = svd(bX)
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test bX == X # check we didn't modify the input

bY = big.(randn(n))
@test isapprox(qr(bX) \ bY, bS \ bY, rtol=1e3*eps(BigFloat))
@test bX == X # check we didn't modify the input

# Note: currently ∄ method for svd(Adjoint{BF,Matrix{BF}})
bXt = Matrix(bX')
bSt = svd(bXt)
@test isapprox(Matrix(bSt), bXt, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bXt), svdvals(X), rtol=1e3*eps())
@test bXt == X' # check we didn't modify the input

X = Float64[1 2 0; 0 1 2; 0 0 0]
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

X = Float64[0 2 0; 0 1 2; 0 0 1]
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input


bD = big.(randn(m))
bX = diagm(0 => bD)
bS = svd(bX)

@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test bS.S == sort(abs.(bD),rev=true)




X = randn(n,m)+im*randn(n,m)
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

bXt = Matrix(bX')
bSt = svd(bXt)
@test isapprox(Matrix(bSt), bXt, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bXt), svdvals(X), rtol=1e3*eps())
@test bXt == X' # check we didn't modify the input


X = ComplexF64[1 2 0; 0 1 2; 0 0 0]
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

X = ComplexF64[0 2 0; 0 1 2; 0 0 1]
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

X = randn(n,1)
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

X = randn(1,n)
bX = big.(X)
bS = svd(bX)
@test isapprox(Matrix(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input
