"""
Script: MetropolisHastings.jl 
Name: Djamil Lakhdar-Hamina
Date: 03/30/2023
Description: 

"""

# TODO: comment 
# TODO: error handling 
# TODO: 1-3-d metropolis hastings 
# TODO: parametrize distribution (not just uniform) (done)
## TODO: optimize code , devectorize
## TODO: make domain independent , add domain option    
    ## TODO: part of this is to make sure walking doesnt go past boundary 
## TODO: make deterministic (seed option)
## TODO: implement adaptive metropolis 

include("NeuralAnsatz.jl")

using Random , Distributions

"""
metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}

Examples
≡≡≡≡≡≡≡≡≡≡

julia> f(x::Real)=ℯ^-x^2
julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)

"""

function metropolis_hastings(ϵ::T0,xmin::T1,xmax::T2,iterations::T3,P::T4;distribution::T5=Uniform,dimension::T6=1,type::T7=Float32,particlesN::T8=1) where {T0<:AbstractFloat, T1<:AbstractFloat, T2<:AbstractFloat,T3<:Integer, T4<:Function,T5<:UnionAll,T6<:Integer,T7<:Type,T8<:Integer}
    i=1
    X=fill(zeros(type,particlesN),iterations)
    x₀=[convert(type,rand(distribution(xmin,xmax))) for _ in 1:particlesN]
    X[i]=x₀
    while i<iterations
        ζ=[convert(type,rand(distribution(-ϵ,ϵ))) for _ in 1:particlesN]
        u=[convert(type,rand()) for _ in 1:particlesN]
        x₁=x₀.+ζ
        accept_index=P.(x₁)./P.(x₀).>=u
        x₀[accept_index]=x₁[accept_index]
        x₀=deepcopy(x₀) 
        i=i+1 
        X[i]=x₀  
    end 
    return X
end 

#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """
# function adaptive_metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function,sample_adapt::Integer,adaptance_rate::Float64)

#     i=1
#     acceptance=0
#     X=zeros(iterations)
#     x₀=rand(Uniform(xmin,xmax))
#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P(x₁)/P(x₀)>=u
#             x₀=x₁
#             acceptance+=1
#             if acceptance//sample_adapt==0  
#         else
#             x₀=x₀
#         end 
#         i+=1 
#         X[i]=x₀
        
#     end 
#     return X
# end 
# end

