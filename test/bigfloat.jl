using GenericSVD
using Base.Test

srand(1)

n,m = 100,20

X = randn(n,m)
bX = big.(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

bY = big.(randn(n))
@test isapprox(qrfact(bX,Val{false}) \ bY, bS \ bY, rtol=1e3*eps(BigFloat))
@test bX == X # check we didn't modify the input

bXt = bX'
bSt = svdfact(bXt)
@test isapprox(full(bSt), bXt, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bXt), svdvals(X), rtol=1e3*eps())
@test bXt == X' # check we didn't modify the input

X = Float64[1 2 0; 0 1 2; 0 0 0]
bX = big.(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

X = Float64[0 2 0; 0 1 2; 0 0 1]
bX = big.(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input


bD = big.(randn(m))
bX = diagm(bD)
bS = svdfact(bX)

@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test bS.S == sort(abs.(bD),rev=true)




X = randn(n,m)+im*randn(n,m)
bX = big.(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

bXt = bX'
bSt = svdfact(bXt)
@test isapprox(full(bSt), bXt, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bXt), svdvals(X), rtol=1e3*eps())
@test bXt == X' # check we didn't modify the input


X = Complex128[1 2 0; 0 1 2; 0 0 0]
bX = big.(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input

X = Complex128[0 2 0; 0 1 2; 0 0 1]
bX = big.(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())
@test bX == X # check we didn't modify the input
