"""
Script: NeuralNetAnsatz
Author:Djamil Lakhdar-Hamina
Date: 06/28/2023
Description:

A script to test quantum monte carlo for calculating ground state 

"""

include("/Users/dlakhdar/physics/GitRepos/Ansatz.jl/src/MetropolisHastings.jl")

using Zygote,Flux

const ϵ=1.5f0
const xmin1=-1.0f0
const xmax1=1.0f0
const λ=.01

struct NeuralAnsatz
    chain::Chain
end
  
# function (m::NeuralAnsatz)(x)
   
#     return exp.(m.chain(x)).^2
# end

function (m::NeuralAnsatz)(x)
   
    return exp.(m.chain(x)-.5f0*x.^2)
end

Flux.@functor NeuralAnsatz

∇(g::Function,𝐱::Vector)=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::Function,𝐱::Vector)=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))
∇(g::NeuralAnsatz,𝐱::Vector)=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::NeuralAnsatz,𝐱::Vector)=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))

chain = Chain(Dense(1, 1000,relu),Dense(1000, 1,relu))
Ψ = NeuralAnsatz(chain)

ϕ(𝐱::Vector)=1/2*𝐱[1]^2
Ĥ(𝐱::Vector, ψ::Function)=-∇²(ψ,𝐱)/2 .+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::Function)=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ) ## ground state energy 

ϕ(𝐱::Vector)=1/2*𝐱[1]^2+λ*𝐱[1]^4
Ĥ(𝐱::Vector, ψ::NeuralAnsatz)=-∇²(ψ,𝐱)/2 .+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::NeuralAnsatz)=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)
ε₀(𝐱::Float64,ψ::NeuralAnsatz)=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)

function Δϵ(N,𝐫,Ψ,Ĥ,ε₀,Θ)
    log∇0=gradient(()->sum(log.(Ψ([𝐫[1]]))),Θ)
    𝒪=[1/Ψ([𝐫[1]])*Ĥ([𝐫[1]],Ψ).*log∇0,ε₀([𝐫[1]],Ψ),log∇0]
    for i in 2:N
        log∇=gradient(()->sum(log.(Ψ([𝐫[i]]))),Θ)
        𝒪[1].+=1/Ψ([𝐫[i]])*Ĥ([𝐫[i]],Ψ).*log∇0 ## return grads objects 
        𝒪[2].+=ε₀([𝐫[i]],Ψ)
        𝒪[3].+=log∇
    end 
    return 2.0f32 .*𝒪[1]  .- 2.0f32 .* ((𝒪[2]./N)[1] .* 𝒪[3]./N)
end 

function Ē(𝐱::Vector, ψ::NeuralAnsatz)
    N=length(𝐱)
    sum_=0
    for i in 1:N
        sum_+=ε₀([𝐱[i]],ψ)[1]
    end 
    1/N*sum_
end 

function train!(epoch,ϵ,xmin1,xmax1,N, Ψ,Ĥ,ε₀,Θ,opt)
    println("training with $N sample points and $epoch iterations...")
    for i in 1:epoch
        𝐫=metropolis_hastings(ϵ,xmin1,xmax1,N,x->sum(Ψ([x]).^2))
        g=Δϵ(N,𝐫,Ψ,Ĥ,ε₀,Θ) ## custom gradient 
        # @show "iteration $i: $(g.params)"
        Flux.update!(opt,Θ,g)
    end 
    println("trained!")
    println("Params: $Θ")
end 

opt=Adam(.1)
N=1000

Θ=Flux.params(Ψ)
epochs=100

function main()
    @time train!(epochs,ϵ,xmin1,xmax1,N, Ψ,Ĥ,ε₀,Θ,opt)
    @time Ē(metropolis_hastings(ϵ,xmin1,xmax1,1000,x->sum(Ψ([x]).^2)),Ψ)
end

main()