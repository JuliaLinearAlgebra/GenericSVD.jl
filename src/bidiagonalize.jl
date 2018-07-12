"""
    PackedUVt <: Factorization{T}

Packed storage of bidiagonalizing QR reflectors `U` and `V'`.


"""
struct PackedUVt{T} <: Factorization{T}
    A::Matrix{T}
end


"""
    bidiagonalize_tall!{T}(A::Matrix{T},B::Bidiagonal)

Bidiagonalize a tall matrix `A` into `B`. Both arguments are overwritten.
"""
function bidiagonalize_tall!(A::Matrix{T},B::Bidiagonal) where T
    m, n = size(A)
    # tall case: assumes m >= n
    # upper bidiagonal
    istriu(B) || throw(ArgumentError("not implemented for lower BiDiag B"))

    for i = 1:n
        x = @view A[i:m, i]
        τi = LinearAlgebra.reflector!(x)
        B.dv[i] = real(A[i,i])
        LinearAlgebra.reflectorApply!(x, τi, @view(A[i:m, i+1:n]))
        A[i,i] = τi # store reflector coefficient in diagonal

        # for Real, this only needs to be n-2
        # needed for Complex to ensure superdiagonal is Real
        if i <= n-1
            x = @view A[i, i+1:n]
            conj!(x)
            τi = LinearAlgebra.reflector!(x)
            B.ev[i] = real(A[i,i+1])
            LinearAlgebra.reflectorApply!(@view(A[i+1:m, i+1:n]), x, τi)
            A[i,i+1] = τi
        end
    end

    B, PackedUVt(A)
end

function bidiagonalize_tall!(A::Adjoint{T2,Matrix{T}}) where {T,T2}
    bidiagonalize_tall!(Matrix(A))
end

function bidiagonalize_tall!(A::Matrix{T}) where T
    m,n = size(A)
    R = real(T)
    B = Bidiagonal(Vector{R}(undef, n), Vector{R}(undef, n - 1), :U)
    bidiagonalize_tall!(A, B)
end

function unpack(P::PackedUVt{T};thin=true) where T
    A = P.A
    m,n = size(A)

    # U = Q_1 ... Q_n I_{m,n}
    w = thin ? n : m
    U = Matrix(one(T)*I,m,w) # eye(T,m,w)
    for i = n:-1:1
        τi = A[i,i]
        x = @view A[i:m, i]
        LinearAlgebra.reflectorApply!(x, τi', @view(U[i:m, i:w]))
    end

    # Vt = P_{n_2} ... P_1
    Vt = Matrix(one(T)*I,n,n) # eye(T,n,n)
    for i = n-1:-1:1
        τi = A[i,i+1]
        x = @view A[i, i+1:n]
        reflectorApply!(@view(Vt[i:n, i+1:n]), x, τi')
    end
    U,Vt
end
