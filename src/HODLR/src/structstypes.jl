
# A null type for the zero function, so I can specialize for cases when I don't actually need to
# assemble matrices or perform any matvecs.
mutable struct ZeroFunction <: Function end

# An abstract type for the level of the HODLR matrix:
abstract type HierLevel end

# An abstract type for the recursive HODLR structure:
abstract type LowRankFact end

# A simple low-rank factorization structure. Used only in the recursive
# structure where this type of storage is easiest.
struct UVt{T} <: LowRankFact
  U::Matrix{T}
  V::Matrix{T}
end

# A recursive HODLR matrix structure. Very simple, and designed for matrices
# that are not symmetric and so you'll probably only want to do matvecs with
# them anyway.
mutable struct RKernelHODLR{T, LF<:LowRankFact} <: AbstractMatrix{T}
  A11::Union{Matrix{T}, RKernelHODLR{T}}
  A22::Union{Matrix{T}, RKernelHODLR{T}}
  A12::LF
  A21::LF
end

# A fixed level, so that the algorith will grow with naive complexity:
mutable struct FixedLevel <: HierLevel
  lv :: Int64
end

# A level that grows with O(logn), so that the level is floor(log2(n)) - lv
mutable struct LogLevel <: HierLevel
  lv :: Int64
end

# A struct to efficiently store low rank matrices of the form (I + M*X*M')(I + M*X*M')'.
mutable struct LowRankW{T<:Number} <: LinearAlgebra.Factorization{T}
  M              ::Matrix{T}
  X              ::LowerTriangular{T, Matrix{T}}
end

# A struct for the symmetric factor of a HODLR matrix.
mutable struct FactorHODLR{T<:Number} <: LinearAlgebra.Factorization{T}
  leafW          :: Vector{LowerTriangular{T, Matrix{T}}}
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
  L              :: Vector{Symmetric{T, Matrix{T}}}            # leaves
  W              :: Union{FactorHODLR{T}, Nothing}             # The symmetric factor
  nys            :: Bool
end

# A block of the derivative of a HODLR matrix. It corresponds to an element of
# KernelHODLR.U or V.
mutable struct DerivativeBlock{T<:Number}
  K1p            :: Matrix{T}
  K1pd           :: Matrix{T}
  Kp2            :: Matrix{T}
  Kp2d           :: Matrix{T}
end

# Similar for the second derivative, although this isn't actually everything.
# This is just all we need for the off-diagonal block beyond what a
# DerivativeBlock already provides.
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
  L              :: Vector{Symmetric{T, Matrix{T}}} 
  B              :: Vector{Vector{DerivativeBlock{T}}}
  S              :: Cholesky{T, Matrix{T}}
  Sj             :: Symmetric{T, Matrix{T}}
end

# A utility struct with all the necessary options for maximum likelihood to keep
# function calls
# somewhat succint.
mutable struct Maxlikopts
  kernfun  :: Function         # The kernel function
  dfuns    :: Vector{Function} # The vector of derivative functions
  d2funs   :: Union{Nothing, Vector{Vector{Function}}} # second derivative functions.
  lvl      :: HierLevel        # The number of dyadic splits of the matrix dimensions. 
  mrnk     :: Int64            # The fixed rank of the off-diagonal blocks.
  saav     :: Vector{Vector{Float64}}   # The SAA vectors
  apll     :: Bool             # Parallel flag for assembly.
  fpll     :: Bool             # Parallel flag for factorization.
  verb     :: Bool             # Verbose flag for optimization path and timings.
  saa_fix  :: Bool             # Flag for whether or not to fix the SAA vectors.
end


mutable struct Maxlikdata{P,T}
  pts_s :: Vector{P}
  dat_s :: Vector{T}
end

