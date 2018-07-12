using GenericSVD
using Test
using Random
using LinearAlgebra


@testset "Miscellany" begin
    include("misc.jl")
end
@testset "BigFloat" begin
    include("bigfloat.jl")
end
@testset "Quaternions" begin
    include("quaternions.jl")
end
