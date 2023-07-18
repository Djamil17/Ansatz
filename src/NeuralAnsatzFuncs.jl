"""
Script: NeuralNetAnsatz
Author:Djamil Lakhdar-Hamina
Date: 06/28/2023
Description:

A script to test quantum monte carlo for calculating ground state 

"""

include("MetropolisHastings.jl")
include("Jackknife.jl")
include("VectorDifferentials.jl")

## TODO : make recursiev for neatness 
function buildChain(;input::T1, width::T2,output::T3=1,depth::T4=1,σ::T5=celu) where {T1<:Integer,T2<:Integer,T3<:Integer,T4<:Integer,T5<:Function}
    
    ## need to build a string
    header="Chain("
    layers="Dense($input,$width,$σ)"
    closing="Dense($width,$output,$σ))"

    if depth==0
       chainString=header*layers*")"
    else
        chainString=layers
        for layer in 1:depth
            if layer==depth
                chainString=header*chainString*","*closing
            else 
                chainString=chainString*","*"Dense($width,$width,$σ)"
            end 
        end 
    end 

    chainExpr=Base.Meta.parse(chainString)        
    return eval(chainExpr)
end 

# function Δϵ(N::Integer,𝐫::Vector{T1},Ψ::T2,Ĥ::T3,ε₀::T4,Θ::Params{Zygote.Buffer{Any, Vector{Any}}}) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function},T3<:Function,T4<:Function}
#     log∇0=gradient(()->sum(log.(Ψ([𝐫[1]]))),Θ)
#     𝒪=[1/Ψ([𝐫[1]])*Ĥ([𝐫[1]],Ψ).*log∇0,ε₀([𝐫[1]],Ψ),log∇0]
#     @inbounds @fastmath @simd for i in 2:N
#         log∇=gradient(()->sum(log.(Ψ([𝐫[i]]))),Θ)
#         𝒪[1].+=1/Ψ([𝐫[i]])*Ĥ([𝐫[i]],Ψ).*log∇0 ## return grads objects 
#         𝒪[2].+=ε₀([𝐫[i]],Ψ)
#         𝒪[3].+=log∇
#     end 
#     return 2.0f32 .*𝒪[1]./N  .- 2.0f32 .* ((𝒪[2]./N)[1] .* 𝒪[3]./N)
# end 

function Δϵ(N::Integer,𝐫::Vector{Vector{T1}},Ψ::T2,Ĥ::T3,ε₀::T4,Θ::Params{Zygote.Buffer{Any, Vector{Any}}}) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function},T3<:Function, T4<:Function}
    log∇0=gradient(()->sum(log.(Ψ(𝐫[1]))),Θ)
    𝒪=[1/Ψ(𝐫[1])*Ĥ(𝐫[1],Ψ).*log∇0,ε₀(𝐫[1],Ψ),log∇0]
    @inbounds @fastmath @simd for i in 2:N
        log∇=gradient(()->sum(log.(Ψ(𝐫[i]))),Θ)
        𝒪[1].+=1/Ψ(𝐫[i])*Ĥ(𝐫[i],Ψ).*log∇0 ## return grads objects 
        𝒪[2].+=ε₀(𝐫[i],Ψ)
        𝒪[3].+=log∇
    end 
    return 2.0f00 .*𝒪[1]./N   .- 2.0f00 .* ((𝒪[2]./N)[1] .* 𝒪[3]./N)
end 

## n particle 
function Ē(𝐱::Vector{Vector{T1}}, ε₀::T2,ψ::T3) where {T1<:AbstractFloat,T2<:Function,T3<:Union{NeuralAnsatz,Function}}
    N=length(𝐱)
    sum_=0
    @inbounds @fastmath @simd for i in 1:N
        sum_+=ε₀(𝐱[i],ψ)[1]
    end 
    1/length(𝐱)*sum_
end 

function train!(epoch::T1,ϵ::T2,xmin1::T3,xmax1::T3,N::T1,Ψ::T4,Ĥ::T5,ε₀::T6,Θ::T7,opt::T8) where {T1<:Integer,T2<:Real, T3<:Real, T4<:Union{NeuralAnsatz,Function}, T5<:Function,T6<:Function,T7<:Params{Zygote.Buffer{Any, Vector{Any}}},T8<:Flux.Optimise.AbstractOptimiser}
    println("training with $N sample points and $epoch iterations...")
    for _ in 1:epoch
        𝐫=metropolis_hastings(Float32,ϵ,xmin1,xmax1,N,x->sum(Ψ([x]).^2))
        ΔE=Δϵ(N,𝐫,Ψ,Ĥ,ε₀,Θ) ## custom gradient 
        Flux.update!(opt,Θ,ΔE)
    end 
    println("trained!")
end 

function train!(particles::T0,epoch::T1,ϵ::T2,xmin::T3,xmax::T3,N::T1,Ψ::T4,Ĥ::T5,ε₀::T6,Θ::T7,opt::T8) where {T0<:Integer,T1<:Integer,T2<:Real, T3<:Real, T4<:Union{NeuralAnsatz,Function}, T5<:Function,T6<:Function,T7<:Params{Zygote.Buffer{Any, Vector{Any}}},T8<:Flux.Optimise.AbstractOptimiser}
    println("training with $N sample points and $epoch iterations...")
    for _ in 1:epoch
        𝐫=metropolis_hastings(
            particles,
            ϵ,
            xmin,xmax,
            N,x->sum(Ψ(x).^2),
            distribution=Uniform
            )

        ΔE=Δϵ(N,𝐫,Ψ,Ĥ,ε₀,Θ) ## custom gradient 
        Flux.update!(opt,Θ,ΔE)
    end 
    println("trained!")
end 

function train!(vmc_problem::vmcProblem,vmc_solution::vmcSolution,opt::Flux.Optimise.AbstractOptimiser,Ĥ::T1,ε₀::T2,Θ::T3) where {T1<:Function,T2<:Function,T3<:Params{Zygote.Buffer{Any, Vector{Any}}}}
    println("training with $(vmc_solution.N) sample points and $(vmc_solution.epoch) iterations...")
    for _ in 1:vmc_solution.epoch
        𝐫=metropolis_hastings(
            vmc_solution.ϵ,
            vmc_problem.Ω[1],vmc_problem.Ω[2],
            vmc_solution.N,
            x->sum(vmc_solution.ψ(x).^2),
            distribution=vmc_solution.distribution,
            spatial_dimension=Int(vmc_problem.spatial_dimension),
            type=Float32,
            particle_number=vmc_problem.particle_number
            )

        ΔE=Δϵ(vmc_solution.N,𝐫,vmc_solution.ψ,Ĥ,ε₀,Θ) ## custom gradient 
        Flux.update!(opt,Θ,ΔE)
    end 
    println("trained!")
end 