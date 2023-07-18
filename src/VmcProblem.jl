
"""

"""

include("VectorDifferentials.jl")
include("NeuralAnsatzFuncs.jl")

function buildNeuralNetAnsatz(input::T0, width::T1,depth::T2,output::T3, σ::T4=celu) where { T0<:Integer,T1<:Integer,T2<:Integer,T3<:Integer,T4<:Function}
    
    chain = buildChain(input=input,width=width,output=output,depth=depth,σ=σ)
    Ψ = NeuralAnsatz(chain)
    return Ψ

end 

# function buildVmcProblem(Ω, ψ::Ansatz, width::T2, depth::T3, ϕ::T4, σ::T5,particle_num::T6) where {T2<:Integer,T3<:Function,T4<:Function, T5<:Integer,T6<:Integer }
#     return vmcProblem(Ω=Ω,Ψ=ψ,width=width,depth=depth,ϕ=ϕ,σ=σ)
# end 

# function buildVmcSolution(ψ::Ansatz,epoch ) where { }
#     return vmcSolution(Ψ=ψ,)
# end 

function runVmcProblem(vmc_problem::vmcProblem,vmc_solution::vmcSolution , target::TF,target_parameters::Tuple{Any}) where {TF<:Function} 
    

    ## TODO: compile these functions as soon as struct is instantiated 
    Ĥ(𝐱::Vector{T1}, ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=-∇²(ψ,𝐱)/2 .+vmc_problem.ϕ(𝐱)*ψ(𝐱)
    ε₀(𝐱::Vector{T1},ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)

    Θ=Flux.params(vmc_solution.ψ) 

    if vmc_solution.hyperparameters==()
        opt=vmc_solution.opt(vmc_solution.η)
    else 
        opt=vmc_solution.opt(vmc_solution.η,vmc_solution.hyperparameters...)
    end 

    train!(
    vmc_problem,vmc_solution,opt,Ĥ,ε₀,Θ
    )

    𝐫=metropolis_hastings(
    vmc_solution.ϵ,
    vmc_problem.Ω[1],
    vmc_problem.Ω[2],
    vmc_solution.N,
    x->sum(vmc_solution.ψ(x).^2),
    distribution=vmc_solution.distribution,
    spatial_dimension=Int(vmc_problem.spatial_dimension),
    type=Float32,
    particle_number=vmc_problem.particle_number
    )

    return target(𝐫, ε₀,target_parameters...)

end