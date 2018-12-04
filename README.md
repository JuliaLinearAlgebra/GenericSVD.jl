# GenericSVD.jl

[![Travis Build Status](https://travis-ci.org/JuliaLinearAlgebra/GenericSVD.jl.svg?branch=master)](https://travis-ci.org/JuliaLinearAlgebra/GenericSVD.jl)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/097ir0q9gpoa6lb6?svg=true)](https://ci.appveyor.com/project/simonbyrne/genericsvd-jl)

Implements Singular Value Decomposition for generic number types, such as `BigFloat`, `Complex{BigFloat}` or [`Quaternion`s](https://github.com/JuliaGeometry/Quaternions.jl). It internally overloads several Base functions such that existing methods (`svd`, `svdfact` and `svdvals`) should work directly.

It uses a Golub-Kahan 2-stage algorithm of bidiagonalization with Householder reflections, followed by an implicit QR with shift.

## Acknowledgements

Based on initial code by Andreas Noack.

## References

* Golub, G. H. and Van Loan, C. F. (2013), "§8.6.3 The SVD Algorithm", *Matrix Computations* (4th ed.), Johns Hopkins University Press, Baltimore, MD, USA.
