for f in ["misc.jl",
          "bigfloat.jl",
          "quaternions.jl"]
    println("Testing $f")
    include(f)
end
