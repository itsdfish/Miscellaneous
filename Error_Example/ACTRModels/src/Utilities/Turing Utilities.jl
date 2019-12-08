import Distributions: rand,logpdf,pdf,estimate

function estimate(model,specs,Nsamples;Nchains=4)
    return reduce(chainscat, pmap(x->sample(model,specs,Nsamples),1:Nchains))
end

function sampleChain(chain)
    parms = (Symbol.(chain.name_map.parameters)...,)
    idx = rand(1:length(chain))
    vals = map(x->chain[x].value[idx],parms)
    return NamedTuple{parms}(vals)
end

function posteriorPredictive(m,chain,f=x->x)
    parms = sampleChain(chain)
    return f(m(parms))
end

function posteriorPredictive(model,chain,Nsamples::Int,f=x->x)
    return map(x->posteriorPredictive(model,chain,f),1:Nsamples)
end

function reduceData(Data)
    U = unique(Data)
    cnt = map(x->count(c->c==x,Data),U)
    newData = NamedTuple[]
    for (u,c) in zip(U,cnt)
        push!(newData,(u...,N=c))
    end
    return newData
end

function logNormParms(μ,σ)
    μ′ = log(μ^2/sqrt(σ^2+μ^2))
    σ′ = sqrt(log(1+σ^2/(μ^2)))
    return μ′,σ′
end

findIndex(actr::ACTR;criteria...) = findIndex(actr.declarative.memory;criteria...)

function findIndex(chunks::Array{<:Chunk,1};criteria...)
    for (i,c) in enumerate(chunks)
        Match(c;criteria...) ? (return i) : nothing
    end
    return -100
end

findIndices(actr::ACTR;criteria...) = findIndices(actr.declarative.memory;criteria...)

function findIndices(chunks::Array{<:Chunk,1};criteria...)
    idx = Int[]
    for (i,c) in enumerate(chunks)
        Match(c;criteria...) ? push!(idx,i) : nothing
    end
    return idx
end

mutable struct LogNormal′{T1,T2} <: ContinuousUnivariateDistribution
  μ::T1
  σ::T2
end

Broadcast.broadcastable(x::LogNormal′) = Ref(x)

Distributions.minimum(d::LogNormal′) = 0.0

Distributions.maximum(d::LogNormal′) = Inf

function rand(d::LogNormal′)
    μ,σ = logNormParms(d.μ,d.σ)
    return rand(LogNormal(μ,σ))
end

function rand(d::LogNormal′,N::Int)
    μ,σ = logNormParms(d.μ,d.σ)
    return rand(LogNormal(μ,σ),N)
end

function logpdf(d::LogNormal′,x::T) where {T<:Real}
    μ,σ = logNormParms(d.μ,d.σ)
    return logpdf(LogNormal(μ,σ),x)
end

function pdf(d::LogNormal′,x::T) where {T<:Real}
    μ,σ = logNormParms(d.μ,d.σ)
    return pdf(LogNormal(μ,σ),x)
end
