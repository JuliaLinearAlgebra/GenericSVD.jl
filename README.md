# GenericSVD.jl

[![Build Status](https://travis-ci.org/simonbyrne/GenericSVD.jl.svg?branch=master)](https://travis-ci.org/simonbyrne/GenericSVD.jl)

Implements Singular Value Decomposition for generic number types, such as `BigFloat` and `Complex{BigFloat}`. It internally overloads several Base functions such that existing methods (`svd`, `svdfact` and `svdvals`) should work directly.

It uses a Golub-Kahan 2-stage algorithm of bidiagonalization with Householder reflections, followed by an implicit QR with shift.
