"""
Script: 

Description:

Short hands for various differential operations in vector calculus 

"""

include("QState.jl")

using Zygote

∇(g::T,𝐱::Vector) where {T<:Union{NeuralAnsatz,Function}}=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::T,𝐱::Vector) where {T<:Union{NeuralAnsatz,Function}}=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))


