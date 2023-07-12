
"""
Script: harmonic_oscillator
Name: Djamil Lakhdar-Hamina
Date: 02/24/2022
Description: 

"""
##TODO : devectorize expressions
## TODO : double derivative gradient
## TODO: AD gradient where a variable is retained  

using Plots,QuadGK, CalculusWithJulia

include("MetropolisHastings.jl")
include("Jackknife.jl")

xmin=0.0
xmax=2.0
ϵ=1.5 ## chosen because I observed acceptance rates, at this value acceptance is aroujnd 60%


