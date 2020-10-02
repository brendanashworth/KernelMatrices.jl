
# The constructor for the HODLR matrix given a KernelMatrix struct. If you
# nystrom=false, the blocks are assembled with the ACA up to tolerance ep or
# rank maxrank (if maxrank>0, otherwise no limit on the permitted rank). If
# nystrom=true, assembles blocks using the Nystrom approximation.
function KernelHODLR(K::KernelMatrix{T,N,A,Fn}, ep::Float64, maxrank::Int64,
                     lvl::HierLevel; nystrom::Bool=false, plel::Bool=false,
                     sorted=false)::KernelHODLR{T} where{T<:Number,N,A,Fn}
  # Warn once about point ordering:
  if !sorted
    @warn "No default point sorting is done for you, and if your points are
    not sorted properly than the approximation can be very poor." maxlog=1
  end

  # Check for symmetry:
  K.x1 == K.x2 || begin
    throw(error(("This function builds symmetric matrices. For non-symmetric matrices,
                 please use the recursive RKernelHODLR, which will only allow matvecs.")))
  end

  # Get the level, leaf indices, and non-leaf indices:
  level, leafinds, nonleafinds = HODLRindices(size(K)[1], lvl)

  # If the Nystrom method was requested, prepare that:
  if nystrom
    maxrank < 2 && error("Please supply a fixed off-diagonal rank that is ≧2.")
    if maxrank >= minimum(map(x->min(x[2]-x[1], x[4]-x[3]), leafinds))
      error("Your nystrom rank is too big. Reduce the HODLR level or nystrom rank.")
    end
    K.x1 == K.x2 || error("Need x1 == x2 for Nystrom approx.")
    nyind = Int64.(round.(LinRange(1, size(K)[1], maxrank)))
    nyker = NystromKernel((x,y)->K.kernel(x,y,K.parms), K.x1[nyind], true)
  end

  level == 0 && @warn "You are using a level-0 HODLR factorization, i.e. dense linear algebra.
         Increase the HODLR level to compress the off-diagonal blocks."

  # Get the leaves in position:
  leaves = mapf(x->Symmetric(full(submatrix(K, x), plel)), leafinds, plel)

  # Get the rest of the decompositions of the non-leaf nodes in place:
  U = Vector{Vector{Matrix{T}}}(undef, level)
  V = Vector{Vector{Matrix{T}}}(undef, level)
  for j in eachindex(U)
    if nystrom
      nonleafinds[j]
      tmpUV = mapf(x->nystrom_uvt(submatrix(K, x), nyker, plel), nonleafinds[j], plel)
    else
      tmpUV = mapf(x->ACA(submatrix(K, x), ep, maxrank), nonleafinds[j], plel)
    end
    U[j] = map(x->x[1], tmpUV)
    V[j] = map(x->x[2], tmpUV)
  end

  return KernelHODLR{T}(ep, level, maxrank, leafinds,
                        nonleafinds, U, V, leaves, nothing, nystrom)

end



# The constructor for the EXACT derivative of a HODLR matrix. It doesn't
# actually require the blocks of the HODLR matrix, but passing it the HODLR
# matrix is convenient to access the information about block boundaries and
# stuff.
function DerivativeHODLR(K::KernelMatrix{T,N,A,Fn}, dfun::Function, HK::KernelHODLR{T};
                         plel::Bool=false) where{T<:Number,N,A,Fn}
  # Check that the call is valid:
  HK.nys || error("This is only valid for Nystrom-block matrices.")

  # Get the landmark point vector, and global S and Sj:
  lndmk  = K.x1[Int64.(round.(LinRange(1, size(K)[1], HK.mrnk)))]
  S      = cholesky(Symmetric(full(KernelMatrix(lndmk, lndmk, K.parms, K.kernel), plel)))
  Sj     = Symmetric(full(KernelMatrix(lndmk, lndmk, K.parms, dfun), plel))

  # Declare the derivative kernel matrix:
  dK     = KernelMatrix(K.x1, K.x2, K.parms, dfun)

  # Get the leaves in position:
  leaves = mapf(x->Symmetric(full(submatrix(dK, x), plel)), HK.leafindices, plel)

  # Get the non-leaves in place:
  B      = Vector{Vector{DerivativeBlock{T}}}(undef, HK.lvl)
  for j in eachindex(B)
    B[j] = mapf(x->DBlock(submatrix(K, x), dfun, lndmk, plel),
                HK.nonleafindices[j], plel)
  end

  return DerivativeHODLR(HK.ep, HK.lvl, HK.leafindices,
                         HK.nonleafindices, leaves, B, S, Sj)
end



# Construct the leaves of the EXACT second derivative of a HODLR matrix.
function SecondDerivativeLeaves(K::KernelMatrix{T,N,A,Fn}, djk::Function,
                                lfi::AbstractVector,
                                plel::Bool=false) where{T<:Number,N,A,Fn}
  d2K    = KernelMatrix(K.x1, K.x2, K.parms, djk)
  return mapf(x->Symmetric(full(submatrix(d2K, x), plel)), lfi, plel)
end



# Construct the off-diagonal blocks of the EXACT second derivative of a HODLR matrix.
function SecondDerivativeBlocks(K::KernelMatrix{T,N,A,Fn}, djk::Function,
                                nlfi::AbstractVector,
                                mrnk::Int64, plel::Bool=false) where{T<:Number,N,A,Fn}
  d2K    = KernelMatrix(K.x1, K.x2, K.parms, djk)
  lndmk  = K.x1[Int64.(round.(LinRange(1, size(K)[1], mrnk)))]
  B      = map(nlf -> mapf(x->SBlock(submatrix(d2K, x), djk, lndmk),
                           nlf, plel), nlfi)
  return Vector{Vector{SecondDerivativeBlock{T}}}(B)
end

# A very simple recursive HODLR matrix structure. At the moment, it only allows
# using the ACA for off-diagonal blocks, so the matrix is not guaranteed to be
# SPSD even if K is. If it is symmetric, though, just use more equipped
# structure anyway. This structure is for when you just need an asymmetric matvec.
#
# I'm also hoping, though, that the new threaded model in julia 1.3 will mean
# that this object can be constructed quite efficiently in massive parallel.
function RKernelHODLR_make(K::KernelMatrix{T,N,A,Fn}, tol::Float64, maxrank::Int64=0,
                      lvl::HierLevel=LogLevel(7)) where{T,N,A,Fn}

  # If the level is LogLevel, call the function again with that FixedLevel:
  if typeof(lvl) <: LogLevel
    lv = FixedLevel(Int64(floor(log2(minimum(size(K))) - lvl.lv)))
    return RKernelHODLR(K, tol, maxrank, lv)
  end

  # Check sizes:
  iszero(lvl.lv) && return full(K)
  len1, len2 = size(K)
  mpt1, mpt2 = Int64(floor(len1/2)), Int64(floor(len2/2))

  # Compute the sub-blocks:
  K11 = submatrix(K, 1,      mpt1, 1,      mpt2)
  K22 = submatrix(K, mpt1+1, len1, mpt2+1, len2)
  K12 = submatrix(K, 1,      mpt1, mpt2+1, len2)
  K21 = submatrix(K, mpt1+1, len1, 1,      mpt2)

  # Factorize the off-diagonal blocks:
  A12 = UVt{T}(ACA(K12, tol, maxrank)...)
  A21 = UVt{T}(ACA(K21, tol, maxrank)...)

  if isone(lvl.lv)
    return RKernelHODLR{T, UVt{T}}(full(K11), full(K22), A12, A21)
  else
    A11 = RKernelHODLR(K11, tol, maxrank, HODLR.FixedLevel(lvl.lv-1))
    A22 = RKernelHODLR(K22, tol, maxrank, HODLR.FixedLevel(lvl.lv-1))
    return RKernelHODLR{T, UVt{T}}(A11, A22, A12, A21)
  end
end

# A wrapper-type function with kwargs to make building this options struct easier.
function maxlikopts(;kernfun, level, rank, saavecs=Vector{Vector}(),
                    fix_saa=true, dfuns=Function[], d2funs=nothing, par_assem=false,
                    par_factor=false, verbose=false)
  return Maxlikopts(kernfun, dfuns, d2funs, level, rank, saavecs,
                    par_assem, par_factor, verbose, fix_saa)
end

function maxlikdata(;pts, data, sortmethod=:KDTree, warn=true)
  if sortmethod == :Nothing
    if warn
      @warn "Not sorting data can adversely affect approximation quality..." maxlog=1
    end
    return Maxlikdata(pts, data)
  elseif sortmethod == :KDTree
    pts_dim    = length(first(pts))
    pts_static = [SVector{pts_dim}(x...) for x in pts]
    pts_s = NearestNeighbors.KDTree(pts_static).data
    return Maxlikdata(pts_s, KernelMatrices.data_reorder(data, pts_static, pts_s))
  #=
  # TODO (cg 2019/12/13 19:29): I could add a Hilbert sort here using
  # geometrical predicates. But I think that code wants some domain restrictions
  # onto [1,2]^d, which gets back to a more serious question from PB about
  # better point enumeration that is more sensitive to rectangular domains.
  elseif sortmethod == :Hilbert
    pts_dim    = length(first(pts))
    in(pts_dim, [2,3]) || throw(error("Hilbert sort only available for d=2 or d=3."))
    pts_pt = [Point(x...) for x in pts]
    hilbertsort!(pts_pt)
    if pts_dim == 2
      pts_s  = [SVector{2}(getx(x), gety(x)) for x in pts_pt]
    else
      pts_s  = [SVector{2}(getx(x), gety(x), getz(x)) for x in pts_pt]
    end
    return Maxlikdata([SVector{pts_dim}
  =#
  else
    throw(error("The options are :Nothing and :KDTree"))
  end
end
