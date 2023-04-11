"""
Script: VMC.jl
Author: Djamil Lakhdar-Hamina
Date: 03/31/2023
"""

using 

include("metropolis_hastings.jl")
include("jackknife.jl")

ϵ=.5
N=1000
Θ=collect(0:.1:1)

trial_wavefunction(x,θ)=√(θ/π)*ℯ^θ*x
X=metropolis_hastings(ϵ,xmin,xmax,N,ψ)

function sgd()

    let
        b, m  = 0.0, 0.0
        for t=1:10_000_000
            η = .00002  # there seems to be an art to picking these steplengths
         
            b, m  =  (b, m) .- η *∇loss(b,m, rand(1:n))
           
        end
           (b=b, m=m)
    end

end 

function vmc(ψ::Function,Ĥ::Function,minimizer::Function,Θ::Array,X::Array)::Tuple(Float64,Float64)
    """
    """

end
    