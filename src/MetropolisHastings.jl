"""
Script: jackknife
Name: Djamil Lakhdar-Hamina
Date: 03/30/2023
Description: 

"""

# TODO: 3-d metropolis hastings 
# TODO: n-d metropolis hastings 
# TODO: parametrize distribution (not just uniform)
# TODO: implement full blown mcmc with metropolis

using Random 
using Distributions

function metropolis_hastings(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}

    """

    """

    i=1
    X=zeros(iterations)
    x₀=rand(Uniform(xmin,xmax))
    X[i]=x₀
    while i<iterations
        ζ=rand(Uniform(-ϵ,ϵ))
        u=rand()
        x₁=x₀+ζ
        if P(x₁)/P(x₀)>= u
            x₀=x₁
        else
            x₀=x₀
        end 
        i+=1 
        X[i]=x₀
        
    end 
    return X
end 

# function metropolis_hastings(ϵ::Float64,Ω::Ω³,iterations::Int64,S::Function)::Array(3,N){Float64}

#     """

#     """

#     i=1
#     X=zeros(iterations)
#     x₀=rand(Uniform(xmin,xmax))
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