

include("NeuralAnsatz.jl")
include("NeuralNetAnsatz.jl")

function buildVmcProblem(Ω::Domain, ψ::Ansatz, width::T2, depth::T2, ϕ::T3, σ::T4, particle_type::T5) where {T2<:Integer,T3<:Function,T5<:Function }
    
    function (m::NeuralAnsatz)(x)
   
        return exp.(m.chain(x))*Ψ
    end 
    
    Flux.@functor NeuralAnsatz

    build
    Ψ=NeuralAnsatz(chain)

    return vmcProblem(Ω,ψ,width,depth,ϕ,σ,particle_type)
end 