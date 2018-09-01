
# The constructor for the HODLR matrix given a KernelMatrix struct. If you nystrom=false, the blocks
# are assembled with the ACA up to tolerance ep or rank maxrank (if maxrank>0, otherwise no limit on
# the permitted rank). If nystrom=true, assembles blocks using the Nystrom approximation.
function KernelHODLR(K::KernelMatrix{T}, ep::Float64=1.0e-10, maxrank::Int64=0, level::Int64=0;
                     nystrom::Bool=false, plel::Bool=false)::KernelHODLR{T} where{T<:Number}

  # Get the level, leaf indices, and non-leaf indices:
  level, leafinds, nonleafinds = HODLRindices(size(K)[1], level)
  nwrk                         = nworkers()

  # If the Nystrom method was requested, prepare that:
  if nystrom
    if maxrank >= minimum(map(x->min(x[2]-x[1], x[4]-x[3]), leafinds))
      error("Your nystrom rank is too big. Reduce the HODLR level or nystrom rank.")
    end
    K.x1 == K.x2 || error("This type of matrix doesn't admit a Nystrom kernel appx. Need x1 == x2")
    nyind = Int64.(round.(linspace(1, size(K)[1], maxrank)))
    nyker = KernelMatrices.NystromKernel(T, K.kernel, K.x1[nyind], K.parms, true)
  end

  # Get the leaves in position:
  leaves = mapf(x->Symmetric(K[x[1]:x[2], x[3]:x[4]]), leafinds, nwrk, plel)

  # Get the rest of the decompositions of the non-leaf nodes in place:
  U = Vector{Vector{Matrix{T}}}(level-1)  
  V = Vector{Vector{Matrix{T}}}(level-1)  
  for j in 1:(level-1)
    if nystrom
      tmpUV = mapf(x->KernelMatrices.nystrom_uvt(nlfisub(K, x), nyker), nonleafinds[j], nwrk, plel)
    else
      tmpUV = mapf(x->KernelMatrices.ACA(nlfisub(K, x), ep, maxrank), nonleafinds[j], nwrk, plel)
    end
    U[j] = map(x->x[1], tmpUV)
    V[j] = map(x->x[2], tmpUV)
  end

  return KernelHODLR{T}(ep, level, maxrank, leafinds, nonleafinds, U, V, leaves, nothing, nystrom)

end



# The constructor for the EXACT derivative of a HODLR matrix. It doesn't actually require the blocks
# of the HODLR matrix, but passing it the HODLR matrix is convenient to access the information about 
# block boundaries and stuff.
function DerivativeHODLR(K::KernelMatrix{T}, dfun::Function, HK::KernelHODLR{T}; 
                         plel::Bool=false) where{T<:Number}
  # Check that the call is valid:
  HK.nys || error("This is only valid for Nystrom-block matrices.")
  nwrk   = nworkers()

  # Get the landmark point vector, and global S and Sj:
  lndmk  = K.x1[Int64.(round.(linspace(1, size(K)[1], HK.mrnk)))]
  S      = cholfact(Symmetric(full(KernelMatrices.KernelMatrix{T}(lndmk, lndmk, K.parms, K.kernel)) + 1.0e-12I))
  Sj     = Symmetric(full(KernelMatrices.KernelMatrix{T}(lndmk, lndmk, K.parms, dfun)))

  # Declare the derivative kernel matrix:
  dK     = KernelMatrix{T}(K.x1, K.x2, K.parms, dfun)

  # Get the leaves in position:
  leaves = mapf(x->Symmetric(dK[x[1]:x[2], x[3]:x[4]]), HK.leafindices, nwrk, plel)

  # Get the non-leaves in place:
  B      = Vector{Vector{DerivativeBlock{T}}}(HK.lvl-1)
  for j in 1:(HK.lvl-1)
    B[j] = mapf(x->DBlock(nlfisub(K, x), dfun, lndmk), HK.nonleafindices[j], nwrk, plel)
  end

  return DerivativeHODLR(HK.ep, HK.lvl, HK.leafindices, HK.nonleafindices, leaves, B, S, Sj)
end



# Construct the leaves of the EXACT second derivative of a HODLR matrix.
function SecondDerivativeLeaves(K::KernelMatrix{T}, djk::Function, lfi::AbstractVector, 
                                plel::Bool=false) where{T<:Number}
  d2K    = KernelMatrices.KernelMatrix{T}(K.x1, K.x2, K.parms, djk)
  return mapf(x->Symmetric(d2K[x[1]:x[2], x[3]:x[4]]), lfi, nworkers(), plel)
end



# Construct the off-diagonal blocks of the EXACT second derivative of a HODLR matrix.
function SecondDerivativeBlocks(K::KernelMatrix{T}, djk::Function, nlfi::AbstractVector,
                                mrnk::Int64, plel::Bool=false) where{T<:Number}
  d2K    = KernelMatrices.KernelMatrix{T}(K.x1, K.x2, K.parms, djk)
  lndmk  = K.x1[Int64.(round.(linspace(1, size(K)[1], mrnk)))]
  B      = map(nlf -> mapf(x->SBlock(nlfisub(d2K, x), djk, lndmk), nlf, nworkers(), plel), nlfi)
  return B
end

