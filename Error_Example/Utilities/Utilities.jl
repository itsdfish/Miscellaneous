"""
Computes the fan values for the stimulus set. Returns a NamedTuple
    for each trial.
"""
function countFan(vals)
    un = (unique(vals)...,)
    uc = map(y->count(x->x==y,vals),un)
    return NamedTuple{un}(uc)
end

"""
Returns fan values for a given person-place pair
"""
function getFan(vals,person,place)
    return (fanPerson=vals[:people][person],fanPlace=vals[:places][place])
end

"""
Computes mean RT for each fan condition
"""
function summarize(vals)
    df = DataFrame(vcat(vals...))
    return by(df,[:fanPerson,:fanPlace,:trial],MeanRT=:rt=>mean)
end
