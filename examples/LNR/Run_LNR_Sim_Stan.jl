using Distributions, CmdStan, Random, Parameters
ProjDir = @__DIR__
cd(ProjDir)
include("LNR.jl")

stream = open("LNR_Model.stan","r")
Model = read(stream,String)
close(stream)

stanmodel = Stanmodel(Sample(save_warmup=false, num_warmup=1000,
  num_samples=1000, thin=1), nchains=1, name="LNR", model=Model,
  printsummary=false, output_format=:mcmcchains)

Random.seed!(343)
Nreps = 50
Nr = 3
Nobs = 10

for i in 1:Nreps
    μ = rand(Normal(0 ,3), Nr)
    σ = rand(Uniform(.2, 1))
    dist = LNR(μ=μ, σ=σ, ϕ=0.0)
    data = rand(dist, Nobs)
    choice = map(x->x[1], data)
    rt = map(x->x[2], data)
    stanInput = Dict("choice"=>choice, "rt"=>rt, "Nr"=>Nr, "N"=>Nobs)
    rc, chain, cnames = stan(stanmodel, stanInput, ProjDir)
end
