VERSION < v"0.7.0-beta2.199" && __precompile__()

module GenericSVD
using LinearAlgebra
import LinearAlgebra: SVD, svd!

include("utils.jl")
include("bidiagonalize.jl")

function svd!(X::AbstractMatrix; full::Bool=false, thin::Union{Bool,Nothing} = nothing)
    if thin != nothing
        @warn "obsolete keyword thin in generic svd!"
        thinx = thin
    else
        thinx = !full
    end
    generic_svdfact!(X; thin=thinx)
end

LinearAlgebra.svdvals!(X::AbstractMatrix) = generic_svdvals!(X)

function generic_svdfact!(X::AbstractMatrix; sorted=true, thin=true)
    m,n = size(X)
    wide = m < n
    if wide
        m,n = n,m
        X = X'
    end
    B,P = bidiagonalize_tall!(X)
    U,Vt = unpack(P,thin=thin)
    U,S,Vt = svd_bidiag!(B,U,Vt)
    # as of Julia v0.7 we need to revert a mysterious transpose here
    Vt=Vt'
    for i = 1:n
        if signbit(S[i])
            S[i] = -S[i]
            for j = 1:n
                Vt[i,j] = -Vt[i,j]
            end
        end
    end
    if sorted
        Idx = sortperm(S,rev=true)
        S = S[Idx]
        U = U[:,Idx]
        Vt = Vt[Idx,:]
    end
    wide ? SVD(Vt',S,U') : SVD(U,S,Vt)
end

function generic_svdvals!(X::AbstractMatrix; sorted=true)
    m,n = size(X)
    if m < n
        X = X'
    end
    B,P = bidiagonalize_tall!(X)
    S = svd_bidiag!(B)
    S .= abs.(S)
    sorted ? sort!(S,rev=true) : S
end


"""
    is_offdiag_approx_zero!(B::Bidiagonal,i,ɛ)

Tests if the element `B[i-1,i]` is approximately zero, using the criteria
```math
    |B_{i-1,i}| ≤ ɛ*(|B_{i-1,i-1}| + |B_{i,i}|)
```
If true, sets the element to exact zero.
"""
function is_offdiag_approx_zero!(B::Bidiagonal,i,ɛ)
    iszero = abs(B.ev[i-1]) ≤ ɛ*(abs(B.dv[i-1]) + abs(B.dv[i]))
    if iszero
        B.ev[i-1] = 0
    end
    iszero
end


"""
    svd_bidiag!(B::Bidiagonal [, U, Vt [, ϵ]])

Compute the SVD of a bidiagonal matrix `B`, via an implicit QR algorithm with shift (known as a Golub-Kahan iterations).

Optional arguments:

 * `U` and `Vt`: orthogonal matrices which pre- and post-multiply `B`, for computing the SVD of a full matrix `X = U*B*Vt` from its bidiagonalized form.

 * `ϵ`: the tolerance for testing zeros of the offdiagonal elements of `B` (see below).

Algorithm:

This proceeds by iteratively finding the lowest strictly-bidiagonal submatrix, i.e. n₁, n₂ such that
```
     [ d ?           ]
     [   d 0         ]
  n₁ [     d e       ]
     [       d e     ]
  n₂ [         d 0   ]
     [           d 0 ]
```
then applying a Golub-Kahan QR iteration.
"""
function svd_bidiag!(B::Bidiagonal{T}, U=nothing, Vt=nothing, ɛ=eps(T)) where T <: Real
    n = size(B, 1)
    if n == 1
        @goto done
    end
    n₂ = n

    maxB = max(maximum(abs, B.dv), maximum(abs, B.ev))

    if istriu(B)
        while true
            @label mainloop

            while is_offdiag_approx_zero!(B,n₂,ɛ)
                n₂ -= 1
                if n₂ == 1
                    @goto done
                end
            end

            n₁ = n₂
            while true
                n₁ -= 1

                # check for diagonal zeros
                if abs(B.dv[n₁]) ≤ ɛ*maxB
                    svd_zerodiag_row!(U,B,n₁,n₂)
                    @goto mainloop
                end
                if n₁ == 1 || is_offdiag_approx_zero!(B,n₁,ɛ)
                    break
                end
            end

            if abs(B.dv[n₂]) ≤ ɛ*maxB
                svd_zerodiag_col!(B,Vt,n₁,n₂)
                @goto mainloop
            end


            d₁ = B.dv[n₂-1]
            d₂ = B.dv[n₂]
            e  = B.ev[n₂-1]

            s₁, s₂ = svdvals2x2(d₁, d₂, e)
            # use singular value closest to sqrt of final element of B'*B
            h = hypot(d₂,e)
            shift = abs(s₁-h) < abs(s₂-h) ? s₁ : s₂
            # avoid infinite loop
            if !all(isfinite.(B))
                if U === nothing
                    return B.dv+NaN
                else
                    return SVD(U .+ NaN, B.dv .+ NaN, Vt .+ NaN)
                end
            end
            svd_gk!(B, U, Vt, n₁, n₂, shift)
        end
    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end
    @label done
    if U === nothing
        return B.dv
    else
        return SVD(U, B.dv, Vt)
    end
end


"""
    svd_zerodiag_row!(U,B,n₁,n₂)

Sets `B[n₁,n₁]` to zero, then zeros out row `n₁` by applying sequential row (left) Givens rotations up to `n₂`, and the corresponding inverse rotations to `U` (preserveing `U*B`.
"""
function svd_zerodiag_row!(U,B,n₁,n₂)
    e = B.ev[n₁]
    B.dv[n₁] = 0 # set to zero
    B.ev[n₁] = 0

    for i = n₁+1:n₂
        # n₁ [0 ,e ] = G * [e ,0 ]
        #    [ ... ]       [ ... ]
        # i  [dᵢ,eᵢ]       [dᵢ,eᵢ]
        dᵢ = B.dv[i]

        G,r = givens(dᵢ,e,i,n₁)
        rmul!(U,adjoint(G))
        B.dv[i] = r # -G.s*e + G.c*dᵢ

        if i < n₂
            eᵢ = B.ev[i]
            e       = G.s*eᵢ
            B.ev[i] = G.c*eᵢ
        end
    end
end


"""
    svd_zerodiag_col!(B::Bidiagonal,Vt,n₁,n₂)

Sets `B[n₂,n₂]` to zero, then zeros out column `n₂` by applying sequential column (right) Givens rotations up to `n₁`, and the corresponding inverse rotations to `Vt` (preserving `B*Vt`).
"""
function svd_zerodiag_col!(B,Vt,n₁,n₂)
    e = B.ev[n₂-1]
    B.dv[n₂] = 0 # set to zero
    B.ev[n₂-1] = 0

    for i = n₂-1:-1:n₁
        #   i      n₂     i      n₂
        #  [eᵢ,...,e ] = [eᵢ,...,0 ] * G'
        #  [dᵢ,...,0 ]   [dᵢ,...,e ]
        dᵢ = B.dv[i]

        G,r = givens(dᵢ,e,i,n₂)
        lmul!(G,Vt)

        B.dv[i] = r # G.c*dᵢ + G.s*e

        if n₁ < i
            eᵢ = B.ev[i-1]
            e         = -G.s*eᵢ
            B.ev[i-1] = G.c*eᵢ
        end
    end
end



"""
    svd_gk!(B::Bidiagonal{T},U,Vt,n₁,n₂,shift) where T <: Real

Applies a Golub-Kahan SVD step (an implicit QR with shift) to the submatrix `B[n₁:n₂,n₁:n₂]`, applying the inverse transformations to `U` and `Vt` (preserving `U*B*Vt`).

A Givens rotation is applied to the top 2x2 matrix, and the resulting "bulge" is "chased" down the diagonal to the bottom of the matrix.
"""
function svd_gk!(B::Bidiagonal{T},U,Vt,n₁,n₂,shift) where T <: Real

    if istriu(B)

        d₁′ = B.dv[n₁]
        e₁′ = B.ev[n₁]
        d₂′ = B.dv[n₁+1]

        G, r = givens(d₁′ - abs2(shift)/d₁′, e₁′, n₁, n₁+1)
        lmul!(G,Vt)

        #  [d₁,e₁] = [d₁′,e₁′] * G'
        #  [b ,d₂]   [0  ,d₂′]


        d₁ =  d₁′*G.c + e₁′*G.s
        e₁ = -d₁′*G.s + e₁′*G.c
        b  =  d₂′*G.s
        d₂ =  d₂′*G.c

        for i = n₁:n₂-2

            #  [. ,e₁′,b′ ] = G * [d₁,e₁,0 ]
            #  [0 ,d₂′,e₂′]       [b ,d₂,e₂]

            e₂ = B.ev[i+1]

            G, r = givens(d₁, b, i, i+1)
            rmul!(U,adjoint(G))

            B.dv[i] =  r # G.c*d₁ + G.s*b

            e₁′ =  G.c*e₁ + G.s*d₂
            d₂′ = -G.s*e₁ + G.c*d₂

            b′  =  G.s*e₂
            e₂′ =  G.c*e₂

            #  [. ,0 ] = [e₁′,b′ ] * G'
            #  [d₁,e₁]   [d₂′,e₂′]
            #  [b ,d₂]   [0  ,d₃′]

            d₃′ = B.dv[i+2]

            G, r = givens(e₁′, b′, i+1, i+2)
            lmul!(G, Vt)

            B.ev[i] = r # e₁′*G.c + b′*G.s

            d₁ =  d₂′*G.c + e₂′*G.s
            e₁ = -d₂′*G.s + e₂′*G.c

            b  = d₃′*G.s
            d₂ = d₃′*G.c
        end

        #  [. ,.] = G * [d₁,e₁]
        #  [0 ,.]       [b ,d₂]

        G, r = givens(d₁,b,n₂-1,n₂)
        rmul!(U, adjoint(G))

        B.dv[n₂-1] =  r # G.c*d₁ + G.s*b

        B.ev[n₂-1] =  G.c*e₁ + G.s*d₂
        B.dv[n₂]   = -G.s*e₁ + G.c*d₂
    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B,U,Vt
end


"""
    svdvals2x2(f, h, g)

The singular values of the matrix
```
B = [ f g ;
      0 h ]
```
(i.e. the sqrt-eigenvalues of `B'*B`).

This is a direct translation of LAPACK [DLAS2](http://www.netlib.org/lapack/explore-html/d8/dfd/dlas2_8f.html).
"""
function svdvals2x2(f, h, g)
    fa = abs(f)
    ga = abs(g)
    ha = abs(h)

    fhmin, fhmax = minmax(fa,ha)

    if fhmin == 0
        ssmin = zero(f)
        if fhmax == 0
            ssmax = zero(f)
        else
            ssmax = max(fhmax,ga)*sqrt(1+(min(fhmax,ga)/max(fhmax,ga))^2)
        end
    else
        if ga < fhmax
            as = 1 + fhmin/fhmax
            at = (fhmax-fhmin)/fhmax
            au = (ga/fhmax)^2
            c = 2/(sqrt(as^2 + au) + sqrt(at^2+au))
            ssmin = fhmin*c
            ssmax = fhmax/c
        else
            au = fhmax / ga
            if au == 0
                ssmin = (fhmin*fhmax)/ga
                ssmax = ga
            else
                as = 1+fhmin/fhmax
                at = (fhmax-fhmin)/fhmax
                c = 1/(sqrt(1 + (as*au)^2) + sqrt(1 + (at*au)^2))
                ssmin = 2((fhmin*c)*au)
                ssmax = ga/(2c)
            end
        end
    end
    ssmin,ssmax
end




end # module
