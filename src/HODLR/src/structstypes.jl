
# A null type for the zero function, so I can specialize for cases when I don't actually need to
# assemble matrices or perform any matvecs.
mutable struct ZeroFunction <: Function end

# A struct to efficiently store low rank matrices of the form (I + M*X*M')(I + M*X*M')'.
mutable struct LowRankW{T<:Number} 
  M              ::Matrix{T}
  X              ::Matrix{T}
end

# A struct for the symmetric factor of a HODLR matrix.
mutable struct FactorHODLR{T<:Number} 
  leafW          :: Vector{Matrix{T}}
  leafWf         :: Vector{LU{T, Matrix{T}}}
  leafWtf        :: Vector{LU{T, Matrix{T}}}
  nonleafW       :: Vector{Vector{LowRankW{T}}}
end

# A HODLR matrix.
mutable struct KernelHODLR{T<:Number}
  ep             :: Float64
  lvl            :: Int64
  mrnk           :: Int64
  leafindices    :: Vector{SVector{4, Int64}}
  nonleafindices :: Vector{Vector{SVector{4, Int64}}}
  U              :: Union{Vector{Vector{Matrix{T}}}, Nothing}  # off-diagonal U 
  V              :: Union{Vector{Vector{Matrix{T}}}, Nothing}  # off-diagonal V
  L              :: Vector{Symmetric{T, Matrix{T}}}         # leaves
  W              :: Union{FactorHODLR{T}, Nothing}             # The symmetric factor, if computed.
  nys            :: Bool
end

# A block of the derivative of a HODLR matrix. It corresponds to an element of KernelHODLR.U or V.
mutable struct DerivativeBlock{T<:Number}
  K1p            :: Matrix{T}
  K1pd           :: Matrix{T}
  Kp2            :: Matrix{T}
  Kp2d           :: Matrix{T}
end

# Similar for the second derivative, although this isn't actually everything. This is just all we
# need for the off-diagonal block beyond what a DerivativeBlock already provides.
mutable struct SecondDerivativeBlock{T<:Number}
  K1pjk          :: Matrix{T}
  Kp2jk          :: Matrix{T}
end

# The derivative of a HODLR matrix.
mutable struct DerivativeHODLR{T<:Number}
  ep             :: Float64
  lvl            :: Int64
  leafindices    :: Vector{SVector{4, Int64}}
  nonleafindices :: Vector{Vector{SVector{4, Int64}}}
  L              :: Vector{Symmetric{T, Matrix{T}}}         # leaves
  B              :: Vector{Vector{DerivativeBlock{T}}}
  S              :: Cholesky{T, Matrix{T}}
  Sj             :: Symmetric{T, Matrix{T}}
end

# A utility struct with all the necessary options for maximum likelihood to keep function calls
# somewhat succint.
mutable struct Maxlikopts
  kernfun  :: Function         # The kernel function
  dfuns    :: Vector{Function} # The vector of derivative functions
  epK      :: Float64          # The pointwise precision for the off-diagonal blocks. Not used for Nystrom method.
  lvl      :: Int64            # The number of dyadic splits of the matrix dimensions. 0 leads to default value.
  mrnk     :: Int64            # The fixed rank of the off-diagonal blocks, with 0 meaning no maximum rank.
  saav     :: Vector{Vector{Float64}}   # The SAA vectors
  apll     :: Bool             # Parallel flag for assembly, which is safe and very beneficial
  fpll     :: Bool             # Parallel flag for factorization, which is less safe and beneficial.
  verb     :: Bool             # Verbose flag to see optimization path and fine-grained times
  saa_fix  :: Bool             # Flag for whether or not to fix the SAA vectors
end
