"""
Script: MetropolisHastings.jl 
Name: Djamil Lakhdar-Hamina
Date: 03/30/2023
Description: 

"""

# TODO: comment 
# TODO: error handling 
# TODO: define types better 
# TODO: 1-3-d metropolis hastings 
# TODO: parametrize distribution (not just uniform)
# TODO: implement full blown mcmc with metropolis
## TODO: optimize code , devectorize
## TODO: make deterministic (seed option)

include("Qstate.jl")

using Random , Distributions

# assuming one particle, one dimension, on a line 

function metropolis_hastings(ϵ::T1,xmin::T2,xmax::T2,iterations::T3,P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
    """

    metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

    """

    i=1
    X=zeros(Float32,iterations)
    x₀=rand(Uniform(xmin,xmax))
    X[i]=x₀
    while i<iterations
        ζ=rand(Uniform(-ϵ,ϵ))
        u=rand()
        x₁=x₀+ζ
        if P([x₁][1])/P([x₀][1])>=u
            x₀=x₁
        else
            x₀=x₀
        end 
        i+=1 
        X[i]=x₀
        
    end 
    return X
end 

# assuming one particle, one dimension, on a line , but choose output type 

function metropolis_hastings(type::Type, ϵ::T1,xmin::T2,xmax::T2,iterations::T3,P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
    """

    metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

    """

    i=1
    X=zeros(type,iterations)
    x₀=rand(Uniform(xmin,xmax))
    X[i]=x₀
    while i<iterations
        ζ=rand(Uniform(-ϵ,ϵ))
        u=rand()
        x₁=x₀+ζ
        if P([x₁][1])/P([x₀][1])>=u
            x₀=x₁
        else
            x₀=x₀
        end 
        i+=1 
        X[i]=x₀
        
    end 
    return X
end 

# assuming one particle, n dimension, on a domain Ω


# function metropolis_hastings(dimension::Number, ϵ::T1,iterations::T3,Ω::Domain, P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

#     i=1
#     X=zeros(iterations,dimension)
#     x₀=rand(Uniform(xmin,xmax))
#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P([x₁])/P([x₀])>=u
#             x₀=x₁
#         else
#             x₀=x₀
#         end 
#         i+=1 
#         X[i]=x₀
        
#     end 
#     return X
# end 

# assuming one particle, n dimension, on a domain Ω, choose output type

# function metropolis_hastings(type::Type, dimension::Integer,ϵ::T1,iterations::T3,Ω::Domain, P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

#     i=1
#     X=zeros(type,iterations,dimension)
#     x₀=rand(Uniform(xmin,xmax))
#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P([x₁])/P([x₀])>=u
#             x₀=x₁
#         else
#             x₀=x₀
#         end 
#         i+=1 
#         X[i]=x₀
        
#     end 
#     return X
# end 

# assuming one particle, 1 dimension, on a domain Ω, choose proposal distribution 
# function metropolis_hastings(Dist::Distributions.Distribution,ϵ::T1,xmin::T2,xmax::T2,iterations::T3,P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

#     i=1
#     X=zeros(Float32,iterations)
#     x₀=rand(Dist(xmin,xmax))
#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P([x₁][1])/P([x₀][1])>=u
#             x₀=x₁
#         else
#             x₀=x₀
#         end 
#         i+=1 
#         X[i]=x₀
        
#     end 
#     return X
# end 

# assuming one particle, 1 dimension, on a domain Ω, choose proposal distribution  and output type
# function metropolis_hastings(Dist::Distributions.Distribution,type::Type, ϵ::T1,xmin::T2,xmax::T2,iterations::T3,P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

#     i=1
#     X=zeros(type,iterations)
#     x₀=rand(Dist(xmin,xmax))
#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P([x₁][1])/P([x₀][1])>=u
#             x₀=x₁
#         else
#             x₀=x₀
#         end 
#         i+=1 
#         X[i]=x₀
        
#     end 
#     return X
# end 

# assuming one particle, n dimension, on a domain Ω, choose proposal distribution  and output type
# function metropolis_hastings(Dist::Distributions.Distribution,type::Type, dimension::Integer, ϵ::T1,Ω::Domain,iterations::T3,P::Function) where {T1<:AbstractFloat, T2<:AbstractFloat, T3<:Integer}
#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

#     i=1
#     X=zeros(type,iterations)
#     x₀=rand(Dist(xmin,xmax))
#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P([x₁][1])/P([x₀][1])>=u
#             x₀=x₁
#         else
#             x₀=x₀
#         end 
#         i+=1 
#         X[i]=x₀
        
#     end 
#     return X
# end 

# assuming n particles, in 1 dimension 

function metropolis_hastings(particlesN::T1, ϵ::T2,xmin::T3,xmax::T3,iterations::T1,P::Function) where {T1<:Integer, T2<:AbstractFloat, T3<:AbstractFloat}
    """
    metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    
    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)

    """

    i=1
    X=fill(zeros(Float32,particlesN),iterations)
    x₀=[convert(Float32,rand(Uniform(xmin,xmax))) for _ in 1:particlesN]
    X[i]=x₀
    while i<iterations
        ζ=[convert(Float32,rand(Uniform(-ϵ,ϵ))) for _ in 1:particlesN]
        u=rand() 
        x₁=x₀.+ζ
        if P(x₁)/P(x₀)>=u
            x₀=x₁
        else
            x₀=x₀
        end 
        i+=1 
        X[i]=x₀    
    end 
    return X
end 

# function adaptive_metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function,sample_adapt::Integer,adaptance_rate::Float64)
#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

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

