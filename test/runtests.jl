using GenericSVD
using Base.Test

srand(1)

a = randexp()
b = randexp()
c = randn()
U = [a c; 0 b]
x,y = GenericSVD.svdvals2x2(a,b,c)

@test sort(sqrt(eigvals(U'*U))) ≈ [x,y]
@test sort(svdvals(U)) ≈ [x,y]

n,m = 100,20

X = randn(n,m)
bX = big(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())


bY = big(randn(n))
@test qrfact(bX,Val{false}) \ bY ≈ bS \ bY

X = randn(n,m)+im*randn(n,m)
bX = big(X)
bS = svdfact(bX)
@test isapprox(full(bS), bX, rtol=1e3*eps(BigFloat))
@test isapprox(svdvals(bX), svdvals(X), rtol=1e3*eps())

