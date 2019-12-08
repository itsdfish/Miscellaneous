"""
Default parameter for the declarative memory module.

* `d`: decay
* `τ`: threshold
* `σ`: noise
* `γ`: maximum associative strength
* `δ`: mismatch penalty
* `ter`: a constant for encoding and responding time
* `blc`: base level constant
* `mmpFun`: mismatch penalty function. Default substracts δ from each non-matching slot value
* `lf:` latency factor parameter
* `bll`: decay and learning on
* `mmp`: mismatch penalty on
* `sa`: spreading activatin on
* `noise`: noise on
* `misc`: NamedTuple of extra parameters
"""
mutable struct Parms{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11}
    d::T1
    τ::T2
    s::T3
    γ::T4
    δ::T5
    blc::T6
    ter::T7
    mmpFun::T8
    lf::T9
    τ′::T10
    bll::Bool
    mmp::Bool
    sa::Bool
    noise::Bool
    misc::T11
end

function Parms(;d=.5, τ=0.0, s=.3, γ=0.0, δ=0.0, blc=0.0, ter=0.0, mmpFun=defaultFun,
    lf=1.0, τ′=zero(typeof(τ)), bll=false, mmp=false, sa=false, noise=false, args...)
    return Parms(d, τ, s, γ, δ, blc, ter, mmpFun, lf, τ′, bll, mmp, sa, noise , args.data)
end


"""
Declarative Memory Chunk
* `N`: number of uses
* `L`: lifetime of chunk
* `created`: time of creation
* `k`: number of chunks in recent set (k=1 is sufficient)
* `dynamic`: slot values are mutable (default: false)
* `slots`: chunk slots. If dynamic, slots are a dictionary. If not dynamic
(default), chunks are an immutable NamedTuple.
* `recent`: time stamps for k recent retrievals
* `lags`: lags for recent retrievals (L - recent)
* `reps`: number of identical chunks. This can be used in simple cases to speed up the code.
* `bl`: baselevel constant added to chunks activation

```
Example:

chunk = Chunk(;created=4.0,person=:hippie,place=:park)
```
"""
mutable struct Chunk{T1,T2}
  N::Int
  L::Float64
  created::Float64
  k::Int
  act::T2
  slots::T1
  reps::Int64
  recent::Array{Float64,1}
  lags::Array{Float64,1}
  bl::T2
end

function Chunk(;N=1, L=1.0, created=0.0, k=1, act=0.0, recent=[0.0],
    reps=0, lags=Float64[], dynamic=false, bl=zero(typeof(act)), slots...)
    if dynamic
        slots = Dict(k=>v for (k,v) in pairs(slots))
        return Chunk(N, L, created, k, act, slots, reps, recent, lags, bl)
    end
    return Chunk(N, L, created, k, act, slots.data, reps, recent, lags, bl)
end

Broadcast.broadcastable(x::Chunk) = Ref(x)

abstract type Mod end

"""
Default parameter for the declarative memory module.

Stores an array of chunks and a parameter object with the following default parameters:

* `d`: decay
* `τ`: threshold
* `σ`: noise
* `γ`: maximum associative strength
* `δ`: mismatch penalty
* `ter`: a constant for encoding and responding time
* `blc`: base level constant
* `mmpFun`: mismatch penalty function. Default substracts δ from each non-matching slot value
* `lf:` latency factor parameter
* `bll`: decay and learning on
* `mmp`: mismatch penalty on
* `sa`: spreading activatin on
* `noise`: noise on
"""
mutable struct Declarative{T1,T2} <: Mod
    memory::Array{T1,1}
    parms::T2
end

function Declarative(;memory=Chunk[], parms...)
    parms′ = Parms(;parms...)
    return  Declarative(memory, parms′)
end

"""
A default function for mismatch penalty. Subtracts δ if
slot does not exist or slot value does not match
"""
function defaultFun(memory, chunk; criteria...)
    slots = chunk.slots
    p = 0.0; δ = memory.parms.δ
    for (k,v) in criteria
        if !(k ∈ keys(slots)) || (slots[k] != v)
            p += δ
        end
     end
     return p
end

Broadcast.broadcastable(x::Declarative) = Ref(x)

"""
Imaginal Module
* `chunk`: chunk in the imaginal module
* `ω`: fan weight. Default is 1.
* `denoms`: cached value for the denominator of the fan calculation
"""
mutable struct Imaginal{T1,T2} <: Mod
    chunk::T1
    ω::T2
    denoms::Vector{Int64}
end

Imaginal(;chunk=Chunk(), ω=1.0, denoms=Int64[]) = Imaginal(chunk, ω, denoms)


"""
ACTR model object
* `declarative`: declarative memory module
* `imaginal`: imaginal memory module
"""
mutable struct ACTR{T1,T2}
    declarative::T1
    imaginal::T2
end

Broadcast.broadcastable(x::ACTR) = Ref(x)

ACTR(;declarative=Declarative(), imaginal=Imaginal()) = ACTR(declarative, imaginal)