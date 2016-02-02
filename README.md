# GenericSVD.jl

[![Build Status](https://travis-ci.org/simonbyrne/GenericSVD.jl.svg?branch=master)](https://travis-ci.org/simonbyrne/GenericSVD.jl)

Implements Singular Value Decomposition for generic number types, such as `BigFloat` and `Complex{BigFloat}`. It internally overloads several Base functions such that existing methods (`svd`, `svdfact` and `svdvals`) should work directly.

It uses a Golub-Kahan 2-stage algorithm of bidiagonalization with Householder reflections, followed by an implicit QR with shift.

## Acknowledgements

Based on initial code by Andreas Noack.

## References

* Golub, G. H. and Van Loan, C. F. (1996), "§8.6.2 The SVD Algorithm", *Matrix Computations* (3rd Ed.), Johns Hopkins University Press, Baltimore, MD, USA.
