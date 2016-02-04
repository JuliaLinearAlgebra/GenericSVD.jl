using GenericSVD
using Quaternions
using Base.Test


n,m = 100,20

X = [quatrand() for i=1:n, j=1:m]

S = svdfact(X)
@test isapprox(full(S), X, rtol=1e3*eps())

Xt = X'
St = svdfact(Xt)
@test isapprox(full(St), Xt, rtol=1e3*eps())
