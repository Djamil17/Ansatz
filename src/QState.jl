"""
Description: 

defines the data structures necessary for arbitrary dimensional ansatz problem 
"""

## TODO: 
## 1. pretty printing 
## 2. better define the type hiearchy distinguish the complex and real parameters 

# abstract type AbstractΩ{T} end  ## abstract concept of a coordinate system for 𝐑 
# abstract type AbstractΘ{T} end ## abstract concept of a parameter space for 𝐋²
# abstract type AbstractQParams{T} end ## abstact concept of a quantum phase space of R and L 

# struct Ω{T<:Real} <:AbstractΩ{T} ## position matrix , each row is a dimension 
#     𝐫::Matrix{T}
# end 

# struct cartesianΩ{T<:Real} <:AbstractΩ{T} ## position matrix , each row is a dimension 
#     𝐱::Vector{T}
#     𝐲::Vector{T}
#     𝐳::Vector{T}
# end

# struct sphericalΩ{T<:Real} <:AbstractΩ{T} ## position matrix , each row is a dimension 
#     𝐫::Vector{T}
#     𝛉::Vector{T}
#     𝛟::Vector{T}
# end 

# struct cylandricalΩ{T<:Real} <:AbstractΩ{T} ## position matrix , each row is a dimension 
#     𝐫::Vector{T}
#     𝛉::Vector{T}
#     𝐳::Vector{T}
# end 

# struct Θ{T<:Real} <: AbstractΘ{T} ## 
#     θ::Vector{T}
# end 

# struct QParams{T<:Real} <: AbstractQParams{T}
#     n::Int ## number of spatial dimensions
#     𝐫::Ω{T}
#     Θ::Θ{T}
# end 

# mutable struct MutQParams{T<:Real} <: AbstractQParams{T}
#     n::Int ## number of spatial dimensions
#     m::Int ## number of non-spatial paramters
#     𝐫::Matrix{T}
#     Θ::Vector{T}
# end 

# struct NeuralAnsatz
#     chain::Chain
# end

# Ansatz=NeuralNetorFunction=Union{NeuralAnsatz,Function}

# abstract type Ansatz end 

# struct VmcProblem{T1<:AbstractFloat}
#     Ω::Domain{T1}
#     ϕ
#     ψ::Ansatz
#     width::Integer
#     depth::Integer
#     σ

#     ϕ(𝐱::Vector)=g^2*a^2-sinh.(a*(𝐱[2]-𝐱[1])).^-2
#     Ĥ(𝐱::Vector, ψ::NeuralAnsatz)=-∇²(ψ,𝐱)/2 .+ϕ(𝐱)*ψ(𝐱)
#     ε₀(𝐱::Vector,ψ::NeuralAnsatz)=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)

# end 

