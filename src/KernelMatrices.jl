
module KernelMatrices

  #@doc read(joinpath(dirname(@__DIR__), "Readme.md"), String) KernelMatrices

  using StaticArrays, SpecialFunctions, SharedArrays, Distributed, LinearAlgebra

  import IterTools
  import LinearAlgebra: mul!

  export KernelMatrix, ACA

  include("structstypes.jl")

  include("baseoverloads.jl")

  include("utils.jl")

  include("factorizations.jl")

  include("covariancefunctions.jl")

  # Its own module:
  include("HODLR/HODLR.jl")

end

