#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
push!(LOAD_PATH, pwd())
using Revise, ACTRModels
include("Utilities/Chunks.jl")
include("Utilities/Stimuli.jl")
include("Utilities/Utilities.jl")
include("Baseline Fan Model.jl")
Random.seed!(8850121)
#######################################################################################
#                                   Generate Data
#######################################################################################
#True value for the base level constant
blc = .3
Nblocks = 5
#Fixed parameters used in the model
parms = (τ=-.5, noise=true, s=.2, ter=.845)
#Generates data for Nblocks. Slots contains the slot-value pairs to populate memory
#stimuli contains the target and foil trials.
temp = simulate(stimuli, slots, parms, Nblocks;blc=blc)
#Forces the data into a concrete type for improved performance
data = vcat(temp...)
#######################################################################################
#                                    Define Model
#######################################################################################
#Creates a model object and passes it to each processor
@everywhere @model model(data, slots, parms) = begin
    #Prior distribution for base level constant
    blc ~ Normal(.3, .15)
    data ~ Fan(blc, parms, slots)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
Nsamples = 2000
Nadapt = 1000
# #Collects sampler configuration options
specs = NUTS(Nadapt, .8)
#Start sampling.
chain = psample(model(data, slots, parms), specs, Nsamples, 4, progress=true)
#######################################################################################
#                                      Summarize
#######################################################################################
println(chain)
