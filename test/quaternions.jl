using GenericSVD
using Quaternions
using Test, Random, LinearAlgebra

srand(1)

n,m = 100,20

X = [quatrand() for i=1:n, j=1:m]

S = svd(X)
@test isapprox(Matrix(S), X, rtol=1e3*eps())

Xt = Matrix(X')
St = svd(Xt)
@test isapprox(Matrix(St), Xt, rtol=1e3*eps())
