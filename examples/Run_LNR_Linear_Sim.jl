using Distributions, Turing, Random, Parameters
ProjDir = @__DIR__
cd(ProjDir)
include("LNR.jl")

@model model(data, x) = begin
    μ = Array{Real,1}(undef,Nr)
    β0 ~ Normal(0, 1)
    β1 ~ Normal(1, 1)
    μ = β0 .+ β1.*x
    s ~ Uniform(0, pi/2)
    σ = tan(s)
    for i in 1:length(data)
        data[i] ~ LNR(μ, σ, 0.0)
    end
end

Random.seed!(3423)
Nreps = 30
error_count = fill(0.0, Nreps)
Nr = 10
Nobs = 10

for i in 1:Nreps
    x = rand(Normal(0, 1), Nr)
    β0 = rand(Normal(0, .5))
    β1 = rand(Normal(1, .5))
    μ = β0 .+ β1.*x
    σ = rand(Uniform(.5, 2))
    dist = LNR(μ=μ, σ=σ, ϕ=0.0)
    data = rand(dist, Nobs)
    chain = sample(model(data, x), NUTS(1000, .8), 2000, discard_adapt = false, progress = false)
    error_count[i] = sum(get(chain,:numerical_error)[1])
end

describe(error_count)
