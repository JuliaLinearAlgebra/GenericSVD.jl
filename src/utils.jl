import Base.LinAlg: reflectorApply!
@inline function reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number) # apply conjugate transpose reflector from right.
    m, n = size(A)
    if length(x) != n
        throw(DimensionMismatch("reflector must have same length as second dimension of matrix"))
    end
    @inbounds begin
        for i = 1:m
            Aiv = A[i, 1]
            for j = 2:n
                Aiv += A[i, j]*x[j]
            end
            Aiv = Aiv*τ
            A[i, 1] -= Aiv
            for j = 2:n
                A[i, j] -= Aiv*x[j]'
            end
        end
    end
    return A
end


import Base: A_mul_B!, A_mul_Bc!, A_ldiv_B!

A_mul_B!(G::LinAlg.Givens, ::Void) = nothing
A_mul_Bc!(::Void, G::LinAlg.Givens) = nothing
