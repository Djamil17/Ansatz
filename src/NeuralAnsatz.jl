"""
Description: 

defines the data structures necessary for arbitrary dimensional ansatz problem 
"""

using Zygote, Flux

## TODO: 
## 1. pretty printing 
## 2. better define the type hiearchy distinguish the complex and real parameters 

struct NeuralAnsatz
    chain::Chain
end

Ansatz=NeuralNetOrFunction=Union{NeuralAnsatz,Function}

##TODO : implement pretty print 

@enum SpatialDimension begin
    one_d=1::Integer
    two_d=2::Integer
    three_d=3::Integer
end

Base.@kwdef struct vmcProblem{T1<:Any,T2<:Integer,T4<:Function}
    Ω::T1 ## where the system is 
    particle_number::T2
    spatial_dimension::SpatialDimension
    ϕ::T4 ## potential of the system 
end 

Base.@kwdef struct vmcSolution{T1<:Integer,T2<:AbstractFloat,T3<:Integer,T5<:AbstractFloat,T7<:Union{Tuple{},Tuple{Any}}}
    ψ::Ansatz ## give a neural net or function
    epoch::T1
    distribution
    ϵ::T2
    N::T3
    opt
    η::T5
    hyperparameters::T7=()

    ## check that the distribution field is a distribution
end 