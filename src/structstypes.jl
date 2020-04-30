
mutable struct KernelMatrix{T, N, A, Fn} <: AbstractPDMat{T}
  x1       ::AbstractVector{A}
  x2       ::AbstractVector{A}
  parms    ::SVector{N, Float64}
  kernel   ::Fn
end

function KernelMatrix(pts, pts2, fn::Function)
  eltype(pts) == eltype(pts2) || error("Eltype of pts and pts2 must agree.")
  A  = eltype(pts)
  T  = typeof(fn(pts[1], pts2[1]))
  return KernelMatrix{T,0,A,typeof(fn)}(pts, pts2, SVector{0, Float64}(), fn)
end

function KernelMatrix(pts, pts2, prms, fn::Function)
  eltype(pts) == eltype(pts2) || error("Eltype of pts and pts2 must agree.")
  A  = eltype(pts)
  T  = typeof(fn(pts[1], pts2[1], prms))
  tp = SVector{length(prms), Float64}(prms)
  return KernelMatrix{T,length(tp),A,typeof(fn)}(pts, pts2, tp, fn)
end

mutable struct NystromKernel{T, A}  <: Function 
  lndmk::Vector{A}
  F::Union{Cholesky{T, Matrix{T}}, BunchKaufman{T, Matrix{T}}}
end

function NystromKernel(kern::Function, landmark, ispd::Bool)::NystromKernel
  T   = typeof(kern(landmark[1], landmark[1]))
  M   = zeros(T, length(landmark), length(landmark))
  for j in eachindex(landmark)
    for k in eachindex(landmark)
      @inbounds M[j,k] = kern(landmark[j], landmark[k])
    end
  end
  F  = ispd ? cholesky!(Symmetric(M)) : bkfact!(Symmetric(M))
  pt = 
  return NystromKernel(landmark, F)
end

