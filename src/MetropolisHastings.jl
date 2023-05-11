"""
Script: MetropolisHastings.jl 
Name: Djamil Lakhdar-Hamina
Date: 03/30/2023
Description: 

"""

# TODO: comment 
# TODO: error handling 
# TODO: 3-d metropolis hastings 
# TODO: n-d metropolis hastings 
# TODO: parametrize distribution (not just uniform)
# TODO: implement full blown mcmc with metropolis
## TODO: optimize code , devectorize

using Random 
using Distributions

# struct Ω³{T}

#     ## a region of 𝐑³ 
#     x_begin::T
#     x_end::T
#     y_begin::T
#     y_end::T
#     z_begin::T 
#     z_end::T 

# end 

function metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)
    """

    metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

    """

    i=1
    X=zeros(iterations)
    x₀=rand(Uniform(xmin,xmax))
    X[i]=x₀
    while i<iterations
        ζ=rand(Uniform(-ϵ,ϵ))
        u=rand()
        x₁=x₀+ζ
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

function metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,θ::Float64,P::Function)
    """

    metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

    """

    i=1
    X=zeros(iterations)
    x₀=rand(Uniform(xmin,xmax))
    X[i]=x₀
    while i<iterations
        ζ=rand(Uniform(-ϵ,ϵ))
        u=rand()
        x₁=x₀+ζ
        if P([x₁,θ])/P([x₀,θ])>=u
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

# function metropolis_hastings(ϵ::Float64,Ω::Ω³,iterations::Int64,S::Function,P::Function)

#     """

#     metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}


#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -1.0,-2.0,10000,f)


#     """

#     i=1
#     m,n=size(Ω)
#     X=zeros(iterations,m)
#     x₀=zeros(m)

#     for j in 1:n
#         append!(x₀,rand(Uniform(Ω[1,j],Ω[2,j])))
#     end  

#     X[i]=x₀
#     while i<iterations
#         ζ=rand(Uniform(-ϵ,ϵ))
#         u=rand()
#         x₁=x₀+ζ
#         if P(x₁,S)/P(x₀,S)>= u
#             x₀=x₁
#         else
#             x₀=x₀
#         end 
#         i=+1 
#         X[i]=x₀
        
#     end 
#     return X
# end 