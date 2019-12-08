using Parameters
import Distributions: logpdf, rand

"""
Distribution constructor for the baseline fan model.

* blc: baselevel constant parameter
* parms: contains all fixed parameters
* slots: slot-value pairs used to populate ACT-R's declarative memory
"""
struct Fan{T1,T2,T3} <: ContinuousUnivariateDistribution
  blc::T1
  parms::T2
  slots::T3
end

#Keyword constructor
Fan(;blc, ter, parms, slots) = Fan(blc, ter, parms, slots)

"""
Computes the log likelihood for the baseline fan model.
"""
function logpdf(d::Fan, data::Array{<:NamedTuple,1})
    LL = computeLL(d.parms, d.slots, data; blc=d.blc)
    return LL
end

"""
Simulates a multiple blocks of trials for the fan experiment.

* blc: baselevel constant parameter
* parms: contains all fixed parameters
* slots: slot-value pairs used to populate ACT-R's declarative memory
* stimuli: vector consisting of slot-value pairs and trial type for the experimental stimuli
"""
function simulate(stimuli, slots, parms, Nblocks; blc)
    #Creates an array of chunks for declarative memory
    chunks = [Chunk(;person=pe,place=pl) for (pe,pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(;memory=chunks, parms..., blc=blc)
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(;declarative=memory)
    data = Array{Array{<:NamedTuple, 1}, 1}(undef,Nblocks)
    for b in 1:Nblocks
        data[b] = simulateBlock(actr, stimuli, slots)
    end
    return vcat(data...)
end

function simulateBlock(actr ,stimuli, slots)
    chunk = Chunk()
    #Extract ter parameter for encoding and motor response
    ter = actr.declarative.parms.ter
    resp = :_
    data = Array{NamedTuple, 1}(undef,length(stimuli))
    i = 0
    #Counts the fan for each person-place pair
    fanCount = map(x->countFan(x),slots)
    for (trial,person,place) in stimuli
        i+=1
        #Randomly select a production rule for person or place retrieval request.
        if rand(Bool)
            chunk = retrieve(actr; person=person)
        else
            chunk = retrieve(actr; place=place)
        end
        #Compute the retrieval time of the retrieved chunk and add ter
        rt = computeRT(actr,chunk) + ter
        if isempty(chunk) || !Match(chunk[1]; person=person, place=place)
            resp = :no
        else
            resp = :yes
        end
        #Get the fan for the person and place
        fan = getFan(fanCount, person, place)
        #Record all of the simulation output for the ith trial
        data[i] = (trial=trial,person=person,place=place,fan...,rt=rt,resp=resp)
    end
    return data
end

"""
Computes the log likelihood of the data

* blc: baselevel constant parameter
* parms: contains all fixed parameters
* slots: slot-value pairs used to populate ACT-R's declarative memory
* stimuli: vector consisting of slot-value pairs and trial type for the experimental stimuli
"""
function computeLL(parms, slots, data; blc)
    act=zero(typeof(blc))
    #Creates an array of chunks for declarative memory
    chunks = [Chunk(person=pe, place=pl, act=act) for (pe,pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(;memory=chunks, blc=blc, parms...)
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(declarative=memory)
    #Don't add noise to activation values
    memory.parms.noise = false
    #Initializes the log likelihood
    LL = 0.0
    #Initialize likelihood array for retrieval production rule mixture
    #Can be typed as a Float64 or Dual for automatic differentiation
    LLs = Array{typeof(blc), 1}(undef,2)
    #Iterate over each trial in data and compute the Loglikelihood based on the response yes, no and
    #rf (retrieval failure)
    #Each response is a mixture of person vs. place retrieval request
    for v in data
        if v.resp == :yes
            LLs[1] = loglike_yes(actr, v.person, v.place, v.rt; person=v.person)
            LLs[2] = loglike_yes(actr, v.person, v.place, v.rt; place=v.place)
            LLs .+= log(.5)
            LL += logsumexp(LLs)
        else
            LLs[1] = loglike_no(actr, v.person, v.place, v.rt; person=v.person)
            LLs[2] = loglike_no(actr, v.person, v.place, v.rt; place=v.place)
            LLs .+= log(.5)
            LL += logsumexp(LLs)
        end
    end
    return LL
end

"""
Computes the simple likelihood of a "yes" response

* actr: actr object
* person: person value of the selected response
* place: place value of the selected response
* rt: observed response time
* request: NamedTuple containing retrieval request (either person or place)
"""
function loglike_yes(actr, person, place, rt; request...)
    #Extract required parameters
    @unpack s,τ,ter=actr.declarative.parms
    #Subset of chunks that match retrieval request
    chunks = getChunk(actr; request...)
    #Find index corresponding to "yes" response, which is the stimulus
    choice = findIndex(chunks; person=person, place=place)
    #Compute the activation for each of the matching chunks
    computeActivation!.(actr, chunks; request...)
    #Collect activation values into a vector
    μ = map(x->x.act, chunks)
    #Add threshold as the last response
    push!(μ, τ)
    #Map the s parameter to the standard deviation for
    #comparability to Lisp ACTR models.
    σ = s*pi/sqrt(3)
    #Create a distribution object for the LogNormal Race model
    dist = LNR(;μ=-μ, σ=σ, ϕ=ter)
    #Compute log likelihood of choice and rt given the parameters.
    return logpdf(dist, choice, rt)
end

"""
Computes the simple likelihood of a "no" response. This function
marginalizes over all of the possible chunks that could be resulted in "no".

* actr: actr object
* person: person value of the selected response
* place: place value of the selected response
* rt: observed response time
* request: NamedTuple containing retrieval request (either person or place)
"""
function loglike_no(actr, person, place, rt; request...)
    #Extract required parameters
    @unpack s,τ,ter,blc=actr.declarative.parms
    #Subset of chunks that match retrieval request
    chunks = getChunk(actr; request...)
    #Compute the activation for each of the matching chunks
    computeActivation!.(actr, chunks; request...)
    #Collect activation values into a vector
    μ = map(x->x.act, chunks)
    #Add threshold as the last response
    push!(μ,τ)
    #Map the s parameter to the standard deviation for
    #comparability to Lisp ACTR models.
    σ = s*pi/sqrt(3)
    #Create a distribution object for the LogNormal Race model
    dist = LNR(;μ=-μ, σ=σ, ϕ=ter)
    #Index of the chunk that represents the stimulus
    idx = findIndex(chunks; person=person, place=place)
    #Initialize likelihood
    N = length(chunks) + 1
    LLs = Array{typeof(blc), 1}()
    #Marginalize over all of the possible chunks that could have lead to the
    #observed response
    for i in 1:N
        #Exclude the chunk representing the stimulus because the response was "no"
        if i != idx
            push!(LLs, logpdf(dist, i, rt))
        end
    end
    return logsumexp(LLs)
end
