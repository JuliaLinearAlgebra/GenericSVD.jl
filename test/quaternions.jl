using GenericSVD
using Quaternions
using Test, Random, LinearAlgebra

Random.seed!(1)

n,m = 100,20

X = [quatrand() for i=1:n, j=1:m]

S = svd(X)
@test isapprox(Matrix(S), X, rtol=1e3*eps())

Xt = Matrix(X')
St = svd(Xt)
@test isapprox(Matrix(St), Xt, rtol=1e3*eps())

@test svdvals([quat(1) quat(0); quat(0) quat(1)]) == [1.0, 1.0]
