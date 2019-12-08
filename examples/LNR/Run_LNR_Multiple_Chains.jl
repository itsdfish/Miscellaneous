using Distributed
addprocs(4)
@everywhere using Distributions, Turing, Random, Parameters, MCMCChains
ProjDir = @__DIR__
cd(ProjDir)
path = pwd()
@everywhere include($path*"/LNR.jl")

@everywhere @model model(data, Nr) = begin
    μ = Array{Real,1}(undef,Nr)
    μ ~ [Normal(0,3)]
    s ~ Uniform(0, pi/2)
    σ = tan(s)
    for i in 1:length(data)
        data[i] ~ LNR(μ, σ, 0.0)
    end
end

Random.seed!(343)
Nreps = 50
Nr = 3
Nobs = 10
μ = rand(Normal(0, 3), Nr)
σ = rand(Uniform(.2, 1))
dist = LNR(μ=μ, σ=σ, ϕ=0.0)
data = rand(dist, Nobs)

function estimate1(model,specs,Nsamples;Nchains=4)
    return reduce(chainscat, pmap(x->sample(model, specs, Nsamples), 1:Nchains))
end


pmap(x->sample(model(data, Nr), NUTS(1000, .8), 2000), 1:4)

reduce(chainscat, pmap(x->sample(model(data, Nr),NUTS(1000, .8), 2000), 1:4))

estimate1(model, NUTS(1000, .8), 2000)
