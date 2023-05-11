"""
Script:AnharmonicOscillator.jl
Author:Djamil Lakhdar-Hamina
Date: 05/04/2023
Description:

"""

using ForwardDiff
using StaticArrays 
import LinearAlgebra:tr,Diagonal

include("MetropolisHastings.jl")
include("Jackknife.jl")
# include("GardientDescent.jl")

struct Ω{T<:Real} ## position matrix , each row is a dimension 
    𝐫::SMatrix{T}
end 

struct Θ{T<:Real} ## 
    θ::Vector{T}
end 

struct QParams{T<:Real}
    n::Int ## number of spatial dimensions
    𝐫::{r{T}}
    Θ::{Θ{T}}
end 

xmin=0.0
xmax=2.0
ϵ=1.5 ## chosen because I observed acceptance rates, at this value acceptance is aroujnd 60%
λ=0.0

∇(g::Function,𝐱)=ForwardDiff.gradient(g,𝐱)
∇²(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)
Δ(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)

∇ᵣ(g::Function,𝐱::Vector)=Diagonal(ForwardDiff.gradient(g,𝐱))[1:end-1]
∇²ᵣ(g::Function,𝐱::Vector)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[1:end-1])
Δᵣ(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)
# ∇θ(g::Function,𝐱)=ForwardDiff.gradient(g,𝐱)
# ∇θ(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)
# Δθ(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)

ψ(𝐱::Vector)=(𝐱[2]/π)^(1/4)*exp(-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2
log_ψ(𝐱::Vector)=log(ψ(𝐱))
ϕ(𝐱::Vector)=1/2*𝐱[1]^2+λ*𝐱[1]^4
# ϕ(𝐱::Vector)=1/2*𝐱[1]^2
Ĥ(𝐱::Vector, ψ::Function)=-∇²ᵣ(ψ,𝐱)/2+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::Function)=ψ(𝐱)^-1*Ĥ(𝐱,ψ) ## ground state energy 

function Δϵ(ψ::Function, Ĥ::Function, ε₀::Function, N::Integer,𝐱::Vector, Aᵢ) 
    𝒪=zeros(3) ## expected values 
    for i in 1:N
        log∇=∇(log_ψ,[𝐱[i], Aᵢ])[end]
        𝒪[1]+=1/ψ([𝐱[i], Aᵢ])*log∇*Ĥ([𝐱[i], Aᵢ],ψ)
        𝒪[2]+=ε₀([𝐱[i], Aᵢ],ψ)
        𝒪[3]+=log∇
    end 
    return 2*𝒪[1]/N-2*𝒪[2]/N*𝒪[3]/N
end 

function Δϵ(ψ::Function, Ĥ::Function, ε₀::Function, N::Integer,Θ::QParams) 
    𝒪=zeros(3) ## expected values 
    for i in 1:N
        log∇=∇(log_ψ,[𝐱[i], Aᵢ])[end]
        𝒪[1]+=1/ψ([𝐱[i], Aᵢ])*log∇*Ĥ([𝐱[i], Aᵢ],ψ)
        𝒪[2]+=ε₀([𝐱[i], Aᵢ],ψ)
        𝒪[3]+=log∇
    end 
    return 2*𝒪[1]/N-2*𝒪[2]/N*𝒪[3]/N
end 

function vmc(N,θ,η,iterations,tolerance,ψ²,Ĥ,ε₀)
    i=0
    Aᵢ=θ
    while i<iterations
        X=metropolis_hastings(ϵ,xmin,xmax,N,Aᵢ,ψ²)
        loss=Δϵ(ψ,Ĥ,ε₀, N,X,Aᵢ)
        # @show loss
        Aᵢ₊₁=Aᵢ-η*loss
        if abs(loss)<tolerance
            break
        end 
        Aᵢ=Aᵢ₊₁
        i+=1
    end 
    return Aᵢ
end 

