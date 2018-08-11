# Avast, LinearAlgebra, prepare to be boarded!
import LinearAlgebra: reflectorApply!

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


import LinearAlgebra: lmul!, rmul!

lmul!(G::LinearAlgebra.Givens{T}, ::Nothing) where T = nothing

# This worked for early betas of 0.7:
# rmul!(::Nothing, Ga::Adjoint{Any,LinearAlgebra.Givens{T}}) where T = nothing

# Now that adjoint of Givens is material:
rmul!(::Nothing, G::LinearAlgebra.Givens{T}) where T = nothing
