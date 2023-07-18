"""
Script: 

Description:

Short hands for various differential operations in vector calculus 

"""

include("NeuralAnsatz.jl")

using LinearAlgebra: Diagonal

∇(g::T1,𝐱::Vector{T2}) where {T1<:Union{NeuralAnsatz,Function}, T2<:AbstractFloat}=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::T1,𝐱::Vector{T2}) where {T1<:Union{NeuralAnsatz,Function},T2<:AbstractFloat}=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))


