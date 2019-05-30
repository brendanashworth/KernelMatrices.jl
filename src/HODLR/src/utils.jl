
mul_t(A, B) = A*transpose(B)
t_mul(A, B) = transpose(A)*B

function mapf(f::Function, v, nwrk::Int64, plel::Bool)
  if plel && nwrk > 1 
    return pmap(f, v, batch_size=max(1, div(length(v), nwrk)))
  else
    return map(f, v)
  end
end

function saa_shuffle!(v::Vector{Vector{T}})::Nothing where{T<:Number}
  for j in eachindex(v)
    rand!(v[j], [-1.0, 1.0])
  end
  nothing
end

function givesaa(len::Int64, sz::Int64; seed::Int64=0)::Vector{Vector{Float64}}
  seed != 0 && Random.seed!(seed)
  vecs = map(x->Array{Float64}(undef, sz), 1:len)
  saa_shuffle!(vecs)
  return vecs
end

function increment!(V::Vector{Int64}, step::Int64)::Nothing
  for k in eachindex(V)
    @inbounds V[k] += step
  end
  nothing
end

function fillall!(target::AbstractVector{T}, src::AbstractVector{T})::Nothing where{T}
  for j in eachindex(target)
    @inbounds setindex!(target, src[j], j)
  end
  nothing
end

# Must be of a Pos-Def Symmetric matrix!
function symfact(A::Symmetric{T, Matrix{T}})::Matrix{T} where{T<:Number}
  factd = eigen(A)
  Out   = factd.vectors
  for j in 1:size(Out, 1)
    @inbounds Out[:,j] .*= sqrt(factd.values[j])
  end
  return Out
end

function Tmatrix(R1::Matrix{T}, R2::Matrix{T})::Symmetric{Float64} where{T<:Number}
  R1R2t = mul_t(R1, R2)
  sz    = size(R1R2t)[1]
  Out   = Symmetric([Matrix(I, sz, sz) R1R2t; transpose(R1R2t) Matrix(I, sz, sz)])
  return Out
end

function lrsymfact(U12::Matrix{T}, U21::Matrix{T})::LowRankW{T} where{T<:Number}
  sz     = size(U12)[2]
  qr12   = qr!(U12)
  qr21   = qr!(U21)
  Q1, R1 = Array(qr12.Q), qr12.R
  Q2, R2 = Array(qr21.Q), qr21.R
  X      = symfact(Tmatrix(R1, R2))-I
  return LowRankW(cat(Q1, Q2, dims=[1,2]), X)
end

function lrx_solterm(W::LowRankW{T}, v::Array{T}) where{T<:Number}
  luf  = lu!(W.X + I)
  Xv   = W.X*v
  return Xv - W.X*(luf\Xv)
end

function lrx_solterm_t(W::LowRankW{T}, v::Array{T}) where{T<:Number}
  luf  = lu!(transpose(W.X) + I)
  Xv   = t_mul(W.X, v)
  return Xv - t_mul(W.X, (luf\Xv))
end

# This function really could use a decent parallel implementation. But I just don't really see how
# to do that considering that I'm mutating memory all over the place.
function invapply!(Wvec::Union{AbstractVector{LowRankW{T}}, AbstractVector{Matrix{T}},
                               AbstractVector{LU{T, Matrix{T}}}}, lvl::Int64,
                   Uvec::Vector{Vector{Matrix{Float64}}}, 
                   Vvec::Vector{Vector{Matrix{Float64}}})::Nothing where{T<:Number}
  jmp = Int64(length(Wvec)/(2*length(Uvec[lvl])))
  ind = collect(1:jmp)
  for j in 1:2*length(Uvec[lvl])
    index = div(j+1, 2)
    if isodd(j)
      @inbounds ldiv!(BDiagonal(Wvec[ind]), Vvec[lvl][index])
    else
      @inbounds ldiv!(BDiagonal(Wvec[ind]), Uvec[lvl][index])
    end
    increment!(ind, jmp)
  end
  nothing
end

function DBlock(K::KernelMatrix{T}, dfun::Function, lndmks::AbstractVector, 
                plel::Bool=false)::DerivativeBlock{T} where{T<:Number}
  K1p  = full(KernelMatrix(K.x1, lndmks, K.parms, K.kernel), plel)
  Kp2  = full(KernelMatrix(lndmks, K.x2, K.parms, K.kernel), plel)
  K1pd = full(KernelMatrix(K.x1, lndmks, K.parms, dfun), plel)
  Kp2d = full(KernelMatrix(lndmks, K.x2, K.parms, dfun), plel)
  return DerivativeBlock(K1p, K1pd, Kp2, Kp2d)
end

function DBlock_mul(B::DerivativeBlock{T}, src::Vector{T}, Kp::Cholesky{T, Matrix{T}},
                    Kpj::Symmetric{T, Matrix{T}})::Vector{T} where{T<:Number}
  out  = B.K1pd*(Kp\(B.Kp2*src))
  out -= B.K1p*(Kp\(Kpj*(Kp\(B.Kp2*src))))
  out += B.K1p*(Kp\(B.Kp2d*src))
  return out
end

function DBlock_mul_t(B::DerivativeBlock{T}, src::Vector{T}, Kp::Cholesky{T, Matrix{T}},
                      Kpj::Symmetric{T, Matrix{T}})::Vector{T} where{T<:Number}
  out  = t_mul(B.Kp2, (Kp\t_mul(B.K1pd, src)))
  out -= t_mul(B.Kp2, Kp\(Kpj*(Kp\(t_mul(B.K1p, src)))))
  out += t_mul(B.Kp2d, Kp\t_mul(B.K1p, src) )
  return out
end

function DBlock_full(B::DerivativeBlock{T}, Kp::Cholesky{T, Matrix{T}},
                     Kpj::Symmetric{T, Matrix{T}})::Matrix{T} where{T<:Number}
  Out  = B.K1pd*(Kp\B.Kp2)
  Out -= B.K1p*(Kp\(Kpj*(Kp\B.Kp2)))
  Out += B.K1p*(Kp\B.Kp2d)
  return Out
end

function SBlock(K::KernelMatrix{T}, djk::Function, lndmks::AbstractVector, 
                plel::Bool=false)::SecondDerivativeBlock{T} where{T<:Number}
  K1pdk = full(KernelMatrix(K.x1, lndmks, K.parms, djk), plel)
  Kp2dk = full(KernelMatrix(lndmks, K.x2, K.parms, djk), plel)
  return SecondDerivativeBlock(K1pdk, Kp2dk)
end

# The many, many parentheses are to ensure that things are done in the reasonable order.
# Like, for matrices A, B, and vector v, we definitely would want to do A*(B*v) instead of 
# (A*B)*v, for example. Unfortunately, the code looks all the more like an unreadable mess for it.
function SDBlock_mul(Bj::DerivativeBlock{T}, Bk::DerivativeBlock{T}, Bjk::SecondDerivativeBlock{T},
                     src::Vector{T}, Sp::Cholesky{T, Matrix{T}}, Spj::Symmetric{T,Matrix{T}},
                     Spk::Symmetric{T, Matrix{T}}, Spjk::Symmetric{T, Matrix{T}}) where{T<:Number}
  # The first term:
  out  = Bjk.K1pjk*(Sp\(Bj.Kp2*src))
  out -= Bj.K1pd*(Sp\(Spk*(Sp\(Bj.Kp2*src))))
  out += Bj.K1pd*(Sp\(Bk.Kp2d*src))
  # The second term:
  out += Bk.K1pd*(Sp\(Spj*(Sp\(Bj.Kp2*src))))
  out -= Bk.K1p*(Sp\(Spk*(Sp\(Spj*(Sp\(Bk.Kp2*src))))))
  out += Bk.K1p*(Sp\(Spjk*(Sp\(Bk.Kp2*src))))
  out -= Bk.K1p*(Sp\(Spj*(Sp\(Spk*(Sp\(Bk.Kp2*src))))))
  out += Bk.K1p*(Sp\(Spj*(Sp\(Bj.Kp2d*src))))
  # The third term:
  out += Bk.K1pd*(Sp\(Bk.Kp2*src))
  out -= Bk.K1p*(Sp\(Spk*(Sp\(Bk.Kp2*src))))
  out += Bk.K1p*(Sp\(Bjk.Kp2jk*src))
  # return it, and thank god that it is over:
  return out
end

function SDBlock_mul_t(Bj::DerivativeBlock{T}, Bk::DerivativeBlock{T},
                       Bjk::SecondDerivativeBlock{T}, src::Vector{T}, Sp::Cholesky{T,Matrix{T}},
                       Spj::Symmetric{T, Matrix{T}}, Spk::Symmetric{T, Matrix{T}},
                       Spjk::Symmetric{T, Matrix{T}}) where{T<:Number}
  # The first term:
  out  = t_mul(Bj.Kp2,  Sp\t_mul(Bjk.K1pjk, src))
  out -= t_mul(Bj.Kp2,  Sp\(Spk*(Sp\t_mul(Bj.K1pd, src))))
  out += t_mul(Bk.Kp2d, Sp\t_mul(Bk.K1pd, src))
  # The second term:
  out += t_mul(Bj.Kp2,  Sp\(Spj*(Sp\t_mul(Bk.K1pd, src))))
  out -= t_mul(Bj.Kp2,  Sp\(Spj*(Sp\(Spk*(Sp\t_mul(Bk.K1p, src))))))
  out += t_mul(Bj.Kp2,  Sp\(Spjk*(Sp\t_mul(Bk.K1p, src))))
  out -= t_mul(Bj.Kp2,  Sp\(Spk*(Sp\(Spj*(Sp\t_mul(Bk.K1p, src))))))
  out += t_mul(Bk.Kp2d, Sp\(Spj*(Sp\t_mul(Bk.K1p, src))))
  # The third term:
  out += t_mul(Bj.Kp2d,   Sp\t_mul(Bk.K1pd, src))
  out -= t_mul(Bj.Kp2d,   Sp\(Spk*(Sp\t_mul(Bk.K1p, src))))
  out += t_mul(Bjk.Kp2jk, Sp\t_mul(Bk.K1p, src))
  # return it, and thank god that it is over:
  return out
end

function Deriv2mul(DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                   D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, D2BL::Vector{Symmetric{T,Matrix{T}}},
                   Sjk::Symmetric{T, Matrix{T}}, src::StridedVector) where{T<:Number}
  # Zero out the target vector:
  target = zeros(eltype(src), length(src))
  # Apply the leaves:
  for j in eachindex(DKj.L)
    c, d = DKj.leafindices[j][3:4]
    target[c:d] += D2BL[j]*src[c:d]
  end
  # Apply the non-leaves:
  for j in eachindex(DKj.B)
    for k in eachindex(DKj.B[j])
      a, b, c, d   = DKj.nonleafindices[j][k]
      target[c:d] += SDBlock_mul_t(DKj.B[j][k], DKk.B[j][k], D2B2[j][k], src[a:b], DKj.S, DKj.Sj, DKk.Sj, Sjk)
      target[a:b] += SDBlock_mul(  DKj.B[j][k], DKk.B[j][k], D2B2[j][k], src[c:d], DKj.S, DKj.Sj, DKk.Sj, Sjk)
    end
  end
  return target
end

# This will be SLOW:
function Deriv2full(DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                    D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, D2BL::Vector{Symmetric{T,Matrix{T}}},
                    Sjk::Symmetric{T, Matrix{T}}, sz::Int64) where{T<:Number}
  out = zeros(T, sz, sz)
  for j in 1:sz
    tmp      = zeros(T, sz)
    tmp[j]   = one(T)
    out[:,j] = Deriv2mul(DKj, DKk, D2B2, D2BL, Sjk, tmp)
  end
  return out
end

