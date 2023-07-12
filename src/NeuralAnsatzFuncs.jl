"""
Script: NeuralNetAnsatz
Author:Djamil Lakhdar-Hamina
Date: 06/28/2023
Description:

A script to test quantum monte carlo for calculating ground state 

"""

include("NeuralAnsatz.jl")
include("MetropolisHastings.jl")
include("Jackknife.jl")
include("VectorDifferentials.jl")

const ϵ=.5f0
const xmin1=-1.0f0
const xmax1=1.0f0
const λ=.01

struct NeuralAnsatz
    chain::Chain
end

Ω=1f-4
α=-0.5f0 

function (m::NeuralAnsatz)(x)
   
    return exp.(Ω*m.chain(sort(x)).-α*(x[1].^2 .+x[2].^2))
end


# function (m::NeuralAnsatz)(x)
   
#     return exp.(m.chain(x).+α*(x.^2))
# end


Flux.@functor NeuralAnsatz

∇(g::T,𝐱::Vector) where {T<:Union{NeuralAnsatz,Function}}=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::T,𝐱::Vector) where {T<:Union{NeuralAnsatz,Function}}=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))

ϕ(𝐱::Vector{T}) where {T<:AbstractFloat}=1/2*𝐱[1]^2+λ*𝐱[1]^4
Ĥ(𝐱::Vector{T1}, ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=-∇²(ψ,𝐱)/2 .+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::T1,ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)
ε₀(𝐱::Vector{T1},ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)

## default one layer NN 
function buildChain(input::T1, width::T1,output::T1,σ::T2) where {T1<:Integer,T2<:Function}
    return Chain(Dense(input, width,σ),Dense(width, output,σ))
end 

## TODO : make recursiev for neatness 
function buildChain(input::T, width::T,depth::T, output::T,σ::T2) where {T<:Integer,T2<:Function}
    
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

function Δϵ(N::Integer,𝐫::Vector{T1},Ψ::T2,Ĥ::T3,ε₀::T4,Θ::Params{Zygote.Buffer{Any, Vector{Any}}}) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function},T3<:Function,T4<:Function}
    log∇0=gradient(()->sum(log.(Ψ([𝐫[1]]))),Θ)
    𝒪=[1/Ψ([𝐫[1]])*Ĥ([𝐫[1]],Ψ).*log∇0,ε₀([𝐫[1]],Ψ),log∇0]
    @simd for i in 2:N
        log∇=gradient(()->sum(log.(Ψ([𝐫[i]]))),Θ)
        𝒪[1].+=1/Ψ([𝐫[i]])*Ĥ([𝐫[i]],Ψ).*log∇0 ## return grads objects 
        𝒪[2].+=ε₀([𝐫[i]],Ψ)
        𝒪[3].+=log∇
    end 
    return 2.0f32 .*𝒪[1]./N  .- 2.0f32 .* ((𝒪[2]./N)[1] .* 𝒪[3]./N)
end 

function Δϵ(N::Integer,𝐫::Vector{Vector{T1}},Ψ::T2,Ĥ::T3,ε₀::T4,Θ::Params{Zygote.Buffer{Any, Vector{Any}}}) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function},T3<:Function, T4<:Function}
    log∇0=gradient(()->sum(log.(Ψ(𝐫[1]))),Θ)
    𝒪=[1/Ψ(𝐫[1])*Ĥ(𝐫[1],Ψ).*log∇0,ε₀(𝐫[1],Ψ),log∇0]
    @simd for i in 2:N
        log∇=gradient(()->sum(log.(Ψ(𝐫[i]))),Θ)
        𝒪[1].+=1/Ψ(𝐫[i])*Ĥ(𝐫[i],Ψ).*log∇0 ## return grads objects 
        𝒪[2].+=ε₀(𝐫[i],Ψ)
        𝒪[3].+=log∇
    end 
    return 2.0f32 .*𝒪[1]./N   .- 2.0f32 .* ((𝒪[2]./N)[1] .* 𝒪[3]./N)
end 


## 1 particle 
function Ē(𝐱::Vector{T1}, ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}
    N=length(𝐱)
    sum_=0
    for i in 1:N
        sum_+=ε₀([𝐱[i]],ψ)[1]
    end 
    1/N*sum_
end 

## n particle 
function Ē(𝐱::Vector{Vector{T1}}, ψ::NeuralAnsatz) where {T1<:AbstractFloat}
    N=length(𝐱)
    sum_=0
    for i in 1:N
        sum_+=ε₀(𝐱[i],ψ)[1]
    end 
    1/N*sum_
end 

function train!(epoch::T1,ϵ::T2,xmin1::T3,xmax1::T3,N::T1,Ψ::T4,Ĥ::T5,ε₀::T6,Θ::T7,opt::T8) where {T1<:Integer,T2<:Real, T3<:Real, T4<:Union{NeuralAnsatz,Function}, T5<:Function,T6<:Function,T7<:Params{Zygote.Buffer{Any, Vector{Any}}},T8<:Flux.Optimise.AbstractOptimiser}
    println("training with $N sample points and $epoch iterations...")
    for i in 1:epoch
        𝐫=metropolis_hastings(Float32,ϵ,xmin1,xmax1,N,x->sum(Ψ([x]).^2))
        ΔE=Δϵ(N,𝐫,Ψ,Ĥ,ε₀,Θ) ## custom gradient 
        # @show "iteration $i: $(g.params)"
        Flux.update!(opt,Θ,ΔE)
    end 
    println("trained!")
    # println("Params: $Θ")
end 

function train!(particles::T0,epoch::T1,ϵ::T2,xmin1::T3,xmax1::T3,N::T1,Ψ::T4,Ĥ::T5,ε₀::T6,Θ::T7,opt::T8) where {T0<:Integer,T1<:Integer,T2<:Real, T3<:Real, T4<:Union{NeuralAnsatz,Function}, T5<:Function,T6<:Function,T7<:Params{Zygote.Buffer{Any, Vector{Any}}},T8<:Flux.Optimise.AbstractOptimiser}
    println("training with $N sample points and $epoch iterations...")
    for i in 1:epoch
        𝐫=metropolis_hastings(particles,ϵ,xmin1,xmax1,N,x->sum(Ψ(x).^2))
        ΔE=Δϵ(N,𝐫,Ψ,Ĥ,ε₀,Θ) ## custom gradient 
        Flux.update!(opt,Θ,ΔE)
    end 
    println("trained!")
end 

"""
Start of program...
"""

opt=Adam(.1)
depth=3
width=100
N=1000
epochs=100
trials=[1,5,10,25,50,100]
depths=[1,2,3]

function test_width()

    variational_info=[]
    for width in trials
        println("Calculating $width ...")
        chain=buildChain(1,width,depth,1,relu)
        Ψ=NeuralAnsatz(chain)
        Θ=Flux.params(Ψ)
        @time train!(epochs,ϵ,xmin1,xmax1,N, Ψ,Ĥ,ε₀,Θ,opt)
        @time E_jack,ΔE=jackknife(metropolis_hastings(Float32, ϵ,xmin1,xmax1,1000,x->sum(Ψ([x]).^2)),ε₀,Ψ)
        push!(variational_info,(width,E_jack,ΔE))
        println("Calculation finished!")

    end 
    return variational_info
end

function test_depth()
    width=10
    variational_info=[]
    for depth in depths
        println("Calculating $depth ...")
        chain=buildChain(1,width,depth,1,relu)
        Ψ=NeuralAnsatz(chain)
        Θ=Flux.params(Ψ)
        @time train!(epochs,ϵ,xmin1,xmax1,N, Ψ,Ĥ,ε₀,Θ,opt)
        @time E_jack,ΔE=jackknife(metropolis_hastings(Float32, ϵ,xmin1,xmax1,1000,x->sum(Ψ([x]).^2)),ε₀,Ψ)
        push!(variational_info,(depth,E_jack,ΔE))
        println("Calculation finished!")

    end 
    return variational_info
end

function main()
    return test_depth(),test_width()
end 

variational_info=main()
depth_E,width_E=variational_info[1],variational_info[2]
E1,uncertainty1=[t[2] for t in depth_E],[t[3] for t in depth_E]
E2,uncertainty2=[t[2] for t in width_E],[t[3] for t in width_E]




