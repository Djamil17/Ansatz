"""
Script: Colagero-Moser.jl 
Name: Djamil Lakhdar-Hamina
Date: 06/30/2023
Description: 

"""

include("../src/NeuralNetAnsatz.jl")

const g=1.0f00
const a=1.0f00

η=.001f0
N=100
ϵ₀=.5
xmin=-1.0
xmax=1.0
n_particles=2

Θ=Flux.params(Ψ)
epochs=1000f0

Ω=η/epochs
α=-0.5f0 

function buildϕ(particleN::T) where {T<:Integer}

    header="g^2*a^2*("
    closing=")"
    middleString=""
    for i in 1:particleN, j in 1:particleN
        if i!=j && i>j 
            if i==particleN && j==particleN-1
            middleString*="sinh.(a*(x[$i]-x[$j])).^-2"
            else 
            middleString*="sinh.(a*(x[$i]-x[$j])).^-2 + "
            end 
        end 
    end 

    chainString=header*middleString*closing
    return @eval(x -> $(Meta.parse(chainString)))
end 

function (m::NeuralAnsatz)(x)
   
    return exp.(Ω*m.chain(sort(x)).+α*(x[1].^2 .+x[2].^2))
end

Flux.@functor NeuralAnsatz

chain = Chain(Dense(2, 5,celu),Dense(5, 1,celu))
Ψ = NeuralAnsatz(chain)

ϕ(𝐱::Vector)=g^2*a^2*sinh.(a*(𝐱[2]-𝐱[1])).^-2
Ĥ(𝐱::Vector, ψ::NeuralAnsatz)=-∇²(ψ,𝐱)/2 .+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::NeuralAnsatz)=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)

opt=Adam(η)

Θ=Flux.params(Ψ)
train!(n_particles,epochs,ϵ₀,xmin,xmax,N,Ψ,Ĥ,ε₀,Θ,opt)
jackknife(metropolis_hastings(n_particles, ϵ,xmin1,xmax1,1000,x->sum(Ψ(x).^2)),ε₀,Ψ)



function main()
    val=[]
    for epoch in [10,100,2000,5000]
        Ω=η/epoch
        chain = Chain(Dense(2, 5,celu),Dense(5, 1,celu))
        Ψ = NeuralAnsatz(chain)
        Θ=Flux.params(Ψ)
        train!(n_particles,epoch,ϵ₀,xmin,xmax,N,Ψ,Ĥ,ε₀,Θ,opt)
        push!(val,Tuple(jackknife(metropolis_hastings(n_particles, ϵ,xmin1,xmax1,1000,x->sum(Ψ(x).^2)),ε₀,Ψ)))
    end
    return val
end 

info=main()
