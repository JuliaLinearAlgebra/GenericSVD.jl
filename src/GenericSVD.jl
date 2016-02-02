module GenericSVD

import Base: SVD, svdvals!, svdfact!

include("utils.jl")
include("bidiagonalize.jl")



function svdfact!(X; sorted=true, thin=true)
    m,n = size(X)
    m >= n || error("Generic SVD requires more rows than columns.")
    B,P = bidiagonalize_tall!(X)
    U,Vt = full(P,thin=thin)
    U,S,Vt = svd!(B,U,Vt)
    for i = 1:n
        if signbit(S[i])
            S[i] = -S[i]
            for j = 1:n
                Vt[i,j] = -Vt[i,j]
            end
        end
    end
    if sorted
        I = sortperm(S,rev=true)
        S = S[I]
        U = U[:,I]
        Vt = Vt[I,:]
    end
    SVD(U,S,Vt)
end

function svdvals!(X; sorted=true)
    B,P = bidiagonalize_tall!(copy(X))
    S = svd!(B)[2]
    for i = eachindex(S)
        if signbit(S[i])
            S[i] = -S[i]
        end
    end
    sorted ? sort!(S,rev=true) : S
end


"""
Tests if the B[i-1,i] element is approximately zero, using the criteria
```math
    |B_{i-1,i}| < ɛ*(|B_{i-1,i-1}| + |B_{i,i}|)
```
"""
offdiag_approx_zero(B::Bidiagonal,i,ɛ) =
    abs(B.ev[i-1]) < ɛ*(abs(B.dv[i-1]) + abs(B.dv[i]))


"""
Generic SVD algorithm:

This finds the lowest strictly-bidiagonal submatrix, i.e. n₁, n₂ such that
```
     [ d ?           ]
     [   d 0         ]
  n₁ [     d e       ]
     [       d e     ]
  n₂ [         d 0   ]
     [           d 0 ]
```
Then applies a Golub-Kahan iteration.
"""
function svd!{T<:Real}(B::Bidiagonal{T}, U=nothing, Vt=nothing, ɛ::T = eps(T))
    n = size(B, 1)
    n₂ = n

    if istriu(B)
        while true
            while offdiag_approx_zero(B,n₂,ɛ)
                n₂ -= 1
                if n₂ == 1
                    return U,B.dv,Vt
                end
            end
            n₁ = n₂ - 1
            while n₁ > 1 && !offdiag_approx_zero(B,n₁,ɛ)
                n₁ -= 1
            end

            # TODO: check for diagonal zeros

            d₁ = B.dv[n₂-1]
            d₂ = B.dv[n₂]
            e  = B.ev[n₂-1]
            
            s₁, s₂ = svdvals2x2(d₁, d₂, e)
            # use singular value closest to
            h = hypot(d₂,e)
            shift = abs(s₁-h) < abs(s₂-h) ? s₁ : s₂            
            svd_gk!(B, U, Vt, n₁, n₂, shift)
        end
    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end
end



"""
Applies a Golub-Kahan SVD step.

A Givens rotation is applied to the top 2x2 matrix, and the resulting "bulge" is "chased" down the diagonal to the bottom of the matrix.
"""
function svd_gk!{T<:Real}(B::Bidiagonal{T},U,Vt,n₁,n₂,shift)

    if istriu(B)
        
        d₁′ = B.dv[n₁]
        e₁′ = B.ev[n₁]
        d₂′ = B.dv[n₁+1]

        G, r = givens(d₁′ - abs2(shift)/d₁′, e₁′, n₁, n₁+1)
        A_mul_B!(G, Vt)

        #  [d₁,e₁] = [d₁′,e₁′] * G
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
            A_mul_Bc!(U, G)

            B.dv[i] =  G.c*d₁ + G.s*b
            
            e₁′ =  G.c*e₁ + G.s*d₂
            d₂′ = -G.s*e₁ + G.c*d₂
            
            b′  =  G.s*e₂
            e₂′ =  G.c*e₂

            #  [. ,0 ] = [e₁′,b′ ] * G
            #  [d₁,e₁]   [d₂′,e₂′]
            #  [b ,d₂]   [0  ,d₃′]

            d₃′ = B.dv[i+2]

            G, r = givens(e₁′, b′, i+1, i+2)
            A_mul_B!(G, Vt)
            
            B.ev[i] = e₁′*G.c + b′*G.s
            
            d₁ =  d₂′*G.c + e₂′*G.s
            e₁ = -d₂′*G.s + e₂′*G.c

            b  = d₃′*G.s
            d₂ = d₃′*G.c
        end

        #  [. ,.] = G * [d₁,e₁] 
        #  [0 ,.]       [b ,d₂]

        G, r = givens(d₁,b,n₂-1,n₂)
        A_mul_Bc!(U, G)
        
        B.dv[n₂-1] =  G.c*d₁ + G.s*b
        
        B.ev[n₂-1] =  G.c*e₁ + G.s*d₂
        B.dv[n₂]   = -G.s*e₁ + G.c*d₂
    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B,U,Vt
end


"""
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

    fhmin = min(fa,ha)
    fhmax = max(fa,ha)
    
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
