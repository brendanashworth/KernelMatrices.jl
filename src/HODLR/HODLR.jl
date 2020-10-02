
module HODLR

  using Pkg, StaticArrays, KernelMatrices, SharedArrays, Transducers
  using Statistics, LinearAlgebra, Distributed, Random

  import NearestNeighbors
  import KernelMatrices: KernelMatrix, submatrix, ACA, full, nystrom_uvt, NystromKernel
  import IterTools:      zip, chain, partition, drop, imap
  import LinearAlgebra:  factorize, mul!, ldiv!, logdet, det, adjoint, transpose, inv, tr, Adjoint

  export KernelHODLR, RKernelHODLR, RKernelHODLR_make, symmetricfactorize!, maxlikopts, LogLevel, FixedLevel, full, maxlikdata

  include("./src/BlockDiagonal.jl")

  include("./src/structstypes.jl")

  include("./src/utils.jl")

  include("./src/baseoverloads.jl")

  include("./src/indices.jl")

  include("./src/constructor.jl")

  include("./src/symfact.jl")

  include("./src/stochasticgradient.jl")

  include("./src/stochastichessian.jl")

  include("./src/maxlikfunctions.jl")

  include("./src/optimization.jl")

  include("./src/exactmaxlik.jl")

  if haskey(Pkg.installed(), "Ipopt")
    using Ipopt
    include("./src/ipoptinterface.jl")
  end
end
