
function Base.size(K::KernelHODLR{T})::Tuple{Int64, Int64} where{T<:Number}
  sz = sum(map(x->size(x,1), K.L))
  return sz, sz
end

function Base.size(K::KernelHODLR{T}, idx::Int64)::Int64 where{T<:Number}
  abs(idx) <= 2 || error("2D array-like thing.")
  sz = sum(map(x->size(x,1), K.L))
  return sz
end

function Base.size(W::LowRankW{T})::Tuple{Int64, Int64} where{T<:Number}
  return size(W.M,1), size(W.M,1)
end

function Base.size(W::LowRankW{T}, j::Int64)::Int64 where{T<:Number}
  return ifelse(j==1, size(W.M,1), size(W.M,2))
end

function Base.size(W::FactorHODLR{T}) where{T<:Number}
  x = sum(x->size(x, 1), W.leafW)
  return (x,x)
end

function det(W::LowRankW{T})::Float64 where{T<:Number}
  return det(I + t_mul(W.M, W.M)*W.X)
end

LinearAlgebra.adjoint(M::LowRankW{T}) where{T<:Number} = Adjoint{T, LowRankW{T}}(M) 

LinearAlgebra.adjoint(A::UVt{T}) where{T<:Number} = Adjoint{T, UVt{T}}(A) 

LinearAlgebra.adjoint(W::FactorHODLR{T}) where{T<:Number} = Adjoint{T, FactorHODLR{T}}(W) 

function mul!(target::StridedArray, W::LowRankW{T}, src::StridedArray) where{T<:Number}
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, W.X*t_mul(W.M, src))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function mul!(target::StridedArray, W::Adjoint{T, LowRankW{T}}, 
              src::StridedArray) where{T<:Number}
  Wp = W.parent
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, Wp.M, Wp.X't_mul(Wp.M, src))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function ldiv!(W::LowRankW{T}, target::StridedArray) where{T<:Number}
  target .-= W.M*lrx_solterm(W, W.M'target)
  return target
end

function ldiv!(W::Adjoint{T,LowRankW{T}}, target::StridedArray) where{T<:Number}
  Wp = W.parent
  target .-= Wp.M*lrx_solterm_t(Wp, Wp.M'target)
  return target
end

function ldiv!(target::StridedArray, W::LowRankW{T}, src::StridedArray) where{T<:Number}
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, -lrx_solterm(W, t_mul(W.M, src)))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function ldiv!(target::StridedArray, W::Adjoint{T,LowRankW{T}}, 
               src::StridedArray) where{T<:Number}
  Wp = W.parent
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, -lrx_solterm_t(Wp, t_mul(Wp.M, src)))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function mul!(target::StridedVector, W::FactorHODLR{T}, 
              src::StridedVector) where{T<:Number}
  # Zero out the target vector, get tmp vector:
  fill!(target, zero(eltype(target)))
  # Apply the nonleafW vectors in the correct order:
  tmp = deepcopy(src)
  for j in length(W.nonleafW):-1:1
    mul!(target, BDiagonal(W.nonleafW[j]), tmp)
    tmp .= target
  end
  # Apply the leaf vectors:
  mul!(tmp, BDiagonal(W.leafW), target)
  target .= tmp
  return target
end

function mul!(target::StridedVector, W::Adjoint{T,FactorHODLR{T}}, 
              src::StridedVector) where{T<:Number}
  Wp = W.parent
  # Zero out the target vector, get tmp vector:
  fill!(target, zero(eltype(target)))
  # Apply the leaf vectors:
  tmp = BDiagonal(Wp.leafW)'src
  # Apply the nonleafW vectors in the correct order:
  for j in eachindex(Wp.nonleafW)
    mul!(target, BDiagonal(Wp.nonleafW[j])', tmp)
    tmp .= target
  end
  return target
end

function ldiv!(target::StridedVector, W::FactorHODLR{T}, 
               src::StridedVector) where{T<:Number}
  # Zero out the target vector, get tmp vector:
  target .= src
  # Apply the leaf vectors:
  ldiv!(BDiagonal(W.leafW), target)
  # Apply the nonleafW vectors in the correct order:
  for j in eachindex(W.nonleafW)
    ldiv!(BDiagonal(W.nonleafW[j]), target)
  end
  return target
end

function ldiv!(target::StridedVector, W::Adjoint{T,FactorHODLR{T}}, 
               src::StridedVector) where{T<:Number}
  Wp = W.parent
  target .= src
  # Apply the nonleafW vectors in the correct order:
  for j in length(Wp.nonleafW):-1:1
    ldiv!(BDiagonal(Wp.nonleafW[j])', target)
  end
  # Apply the leaf vectors:
  ldiv!(BDiagonal(Wp.leafW)', target)
  return target
end

function mul!(target::StridedVector, K::KernelHODLR{T}, 
              src::StridedVector) where{T<:Number}
  if K.W == nothing
    # Zero out the target vector:
    fill!(target, zero(eltype(target)))
    # Apply the leaves:
    for j in eachindex(K.L)
      c, d = K.leafindices[j][3:4]
      mul!(view(target, c:d), K.L[j], src[c:d])
    end
    # Apply the non-leaves:
    for j in eachindex(K.U)
      for k in eachindex(K.U[j])
        a, b, c, d   = K.nonleafindices[j][k]
        target[c:d] += K.V[j][k]*(transpose(K.U[j][k])*src[a:b])
        target[a:b] += K.U[j][k]*(transpose(K.V[j][k])*src[c:d])
      end
    end
  else
    # Zero out the target vector:
    fill!(target, zero(eltype(target)))
    # Multiple by W^{T}, then by W:
    tmp = Array{eltype(target)}(undef, length(target))
    mul!(tmp, adjoint(K.W), src)
    mul!(target, K.W, tmp)
  end
  return target
end

function ldiv!(target::StridedVector, K::KernelHODLR{T}, 
               src::StridedVector) where{T<:Number}
  if K.W == nothing
    error("No solves without factorization.")
  else
    # divide by W, then by W^{T}:
    tmp = Array{eltype(target)}(undef, length(target))
    ldiv!(tmp, K.W, src)
    ldiv!(target, adjoint(K.W), tmp)
  end
  return target
end

function logdet(K::KernelHODLR{T})::Float64 where{T<:Number}
  if K.W == nothing
    error("No logdet without factorization.")
  else
    logdett = 0.0
    for j in eachindex(K.L)
      @inbounds logdett += logdet(factorize(K.L[j]))
    end
    for j in eachindex(K.W.nonleafW)
      for k in eachindex(K.W.nonleafW[j])
         @inbounds logdett += 2.0*log(abs(det(K.W.nonleafW[j][k])))
      end
    end
    return logdett
  end
end

function Base.size(DK::DerivativeHODLR{T})::Tuple{Int64, Int64} where{T<:Number}
  sz = sum(map(x->size(x,1), DK.L))
  return sz, sz
end

function mul!(target::StridedVector, DK::DerivativeHODLR{T}, 
              src::StridedVector) where{T<:Number}
  # Zero out the target vector:
  fill!(target, zero(eltype(target)))
  # Apply the leaves:
  for j in eachindex(DK.L)
    c, d = DK.leafindices[j][3:4]
    target[c:d] += DK.L[j]*src[c:d]
  end
  # Apply the non-leaves:
  for j in eachindex(DK.B)
    for k in eachindex(DK.B[j])
      a, b, c, d   = DK.nonleafindices[j][k]
      target[c:d] += DBlock_mul_t(DK.B[j][k], src[a:b], DK.S, DK.Sj)
      target[a:b] += DBlock_mul(DK.B[j][k],   src[c:d], DK.S, DK.Sj)
    end
  end
  return target
end

##
#
# Convenience overloads:
#
##

function LinearAlgebra.:*(K::KernelHODLR{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(src))
  return mul!(target, K, src)
end

function LinearAlgebra.:*(W::FactorHODLR{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(src))
  return mul!(target, W, src)
end

function LinearAlgebra.:\(K::KernelHODLR{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(src))
  return ldiv!(target, K, src)
end

function LinearAlgebra.:\(K::KernelHODLR{T}, src::Matrix{T})::Matrix{T} where{T<:Number}
  target = similar(src)
  for j in 1:size(src, 2)
    ldiv!(view(target, :, j), K, view(src, :, j))
  end
  return target
end

function LinearAlgebra.:*(W::LowRankW{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(src))
  return mul!(target, W, src)
end

function LinearAlgebra.:*(W::LowRankW{T}, src::Matrix{T})::Matrix{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  return mul!(target, W, src)
end

function LinearAlgebra.:*(DK::DerivativeHODLR{T}, 
                          src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  return mul!(target, DK, src)
end

function LinearAlgebra.:\(W::LowRankW{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(src))
  return ldiv!(target, W, src)
end

function LinearAlgebra.:\(W::LowRankW{T}, src::Matrix{T})::Matrix{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  return ldiv!(target, W, src)
end

@inline Base.size(UV::UVt{T}) where{T} = (size(UV.U, 1), size(UV.V, 2))

@inline Base.size(UV::UVt{T}, j) where{T} = size(UV)[j]

@inline Base.:*(UV::UVt{T}, v::Vector{T}) where{T} = UV.U*(UV.V'v)

@inline Base.size(R::RKernelHODLR) = size(R.A11) .+ size(R.A22)

@inline Base.size(R::RKernelHODLR, j::Int64) = size(R.A11, j) + size(R.A22, j)

function Base.:*(R::RKernelHODLR, V::Vector)
  ix1 = 1:size(R.A11, 2)
  ix2 = (size(R.A11, 2)+1):size(R, 2)
  return vcat(R.A11*V[ix1] + R.A12*V[ix2], R.A21*V[ix1] + R.A22*V[ix2])
end

# Not technically base overloads, but in the same spirit:

function full(M::LowRankW{T})::Matrix{T} where{T<:Number}
  return I + mul_t(M.M*M.X, M.M)
end

function full(K::KernelHODLR{T})::Matrix{T} where{T<:Number}
  Out = Array{T}(undef, size(K))
  for (j,pt) in enumerate(K.leafindices)
    Out[pt[1]:pt[2], pt[3]:pt[4]] = K.L[j]
  end
  for lev in eachindex(K.U)
    for j in eachindex(K.U[lev])
      a, b, c, d    = K.nonleafindices[lev][j]
      Out[a:b, c:d] = mul_t(K.U[lev][j], K.V[lev][j])
      Out[c:d, a:b] = mul_t(K.V[lev][j], K.U[lev][j])
    end
  end
  return Out
end

# VERY computationally inefficient. This really is only for testing.
function full(W::FactorHODLR{T})::Matrix{T} where{T<:Number}
  Out = cat(W.leafW..., dims=[1,2])
  # Multiply the nonleaves:
  for j in 1:length(W.nonleafW)
    Out = Out*cat(full.(W.nonleafW[j])..., dims=[1,2])
  end
  return Out
end

function full(DK::DerivativeHODLR{T})::Matrix{T} where{T<:Number}
  Out = zeros(T, size(DK))
  for (j,pt) in enumerate(DK.leafindices)
    Out[pt[1]:pt[2], pt[3]:pt[4]] = DK.L[j]
  end
  for lev in eachindex(DK.B)
    for j in eachindex(DK.B[lev])
      a, b, c, d     = DK.nonleafindices[lev][j]
      Out[a:b, c:d]  = DBlock_full(DK.B[lev][j], DK.S, DK.Sj)
      Out[c:d, a:b]  = transpose(Out[a:b, c:d])
    end
  end
  return Out
end

@inline full(UV::UVt{T}) where{T} = UV.U*UV.V'

@inline full(M::Matrix{T}) where{T} = M

full(R::RKernelHODLR) = [full(R.A11) full(R.A12) ; full(R.A21) full(R.A22)]

