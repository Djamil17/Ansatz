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

struct vmcProblem{T1<:AbstractFloat,T2<:Integer}
    Ω::Domain{T1}
    ψ::Ansatz
    width::T2
    depth::T2
    ϕ
    σ
    particle_type
end 

function buildVmcProblem(Ω::Domain, ψ::Ansatz, width::T2, depth::T2, ϕ::T3, σ::T4,particle_num::T5, particle_type::T6) where {T2<:Integer,T3<:Function,T5<:Integer,T6<:Integer }
    
    function (m::NeuralAnsatz)(x)
   
        return exp.(m.chain(x))*Ψ
    end 
    
    Flux.@functor NeuralAnsatz

    chain=buildChain(particle_num,width,depth,σ)
    Ψ=NeuralAnsatz(chain)

    return vmcProblem(Ω,ψ,width,depth,ϕ,σ,particle_type)
end 





