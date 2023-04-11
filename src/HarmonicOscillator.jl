
"""
Script: harmonic_oscillator
Name: Djamil Lakhdar-Hamina
Date: 02/24/2022
Description: 

"""

using Plots
using QuadGK
using LaTeXStrings
using CalculusWithJulia

include("MetropolisHastings.jl")
include("Jackknife.jl")
include("GardientDescent.jl")

xmin=0.0
xmax=2.0
ϵ=.5

# test on arbitrary S(x)
# ϵ=.1
# N=100000
# X=metropolis_hastings(ϵ,xmin,xmax,N,S)
# y=ℯ.^-S.(X)
# Z=quadgk(f, -Inf, Inf, rtol=1e-3)[1]
# Ē=1/N*sum(X.^2)
# histogram(X,normed=true,label="binned values")
# scatter!(X,y./Z,label="pi*e^-S(x)")
# vline!([Ē],color="red",label="<x^2>")
# title!("Sampling of pi(x)ℯ^(x²-x⁴) : ϵ=.1 , N=100000")
# xlabel!(L"x")
# ylabel!(L"pi(x) ℯ^-S(x)")

## test on gaussian
# ϵ=.5
# gaussian(x)=x^2
# g(x)=exp(-x^2)
# X=metropolis_hastings(ϵ,xmin,xmax,N,gaussian)
# y=g.(X)
# Z=quadgk(g, -Inf, Inf, rtol=1e-3)[1]
# Ē=sum(1/N*X.^2)
# histogram(X,normed=true)
# scatter!(X,y./Z)
# vline!([Ē],color="red")
# title!("Sampling of pi(x)ℯ^(x²-x⁴) : ϵ=.1 , N=100000")
# xlabel!(L"x")
# ylabel!(L"pi(x) ℯ^-S(x)")

## test on multidimensional gaussian in x,y,z coordinates  
# gaussian_3d(x,y,z)=x^2+y^2+z^2
# ## test on maxwell boltzmann distribution χ² distribution
# ħ=1.054e-34
# m=1
# A=.7
# ω=.7ħ
# N=1000000

function stupid_descent(N::Int,As::Array{Float64,1},block_num::Int)

    if block_num>N
        error("The block number must be greater than the number of iterations")
    end 

    Es=[]
    Δs=[]
    for A in As 
        ψ(x)=(A/(pi))^(1/4)*exp(-A*x^2)
        ψ²(x)=(A/(pi))^(1/2)*exp(-2A*x^2)
        Ĥ(x)=1/2*(1/ψ(x)* -ψ''(x)+x^2)
        X=metropolis_hastings(.5,0.0,1.0,N,ψ²)
        # Ē0=1/N*sum(Ĥ.(X))
        X_new=blocking(X,block_num)
        Ē0,ΔĒ=jackknife(X_new,Ĥ)
        append!(Es,Ē0)
        append!(Δs,ΔĒ)
    end 

    return  Es, Δs
    
end 

function vmc_harmonic(N,θ,η,iterations,tolerance)
    i=0
    Aᵢ=θ
    while i<iterations
        ψ(x)=(Aᵢ/(pi))^(1/4)*exp(-Aᵢ*x^2)
        ψ²(x)=(Aᵢ/(pi))^(1/2)*exp(-2Aᵢ*x^2)
        Ĥ(x)=1/2*(1/ψ(x)* -ψ''(x)+x^2)
        X=metropolis_hastings(ϵ,0.0,5.0,N,ψ²)
        loss=2 * ( 1/N*sum(  ((-4Aᵢ.*X.^2 .-1)/4Aᵢ) .* (-(Aᵢ^5/π)^(1/4).*(2Aᵢ*X.^2 .-1)+ 1/2*X.^2)  )  - (1/N*sum(Ĥ.(X))*1/N*sum((-4Aᵢ*X.^2 .-1)/4Aᵢ) )   )
        Aᵢ₊₁= Aᵢ-η* loss
        if abs(loss)<tolerance
            break
        end 
        Aᵢ=Aᵢ₊₁
        i+=1
    end 
    return Aᵢ
end 

## compute stupid descent and good descent
Ē=.5

As=collect(.1:.1:2)
n=50000
Es, Δs=stupid_descent(n,As,10)
min_e=minimum(Es)
min_A=As[argmin(Es)]

A_vmc=vmc_harmonic(50000,.2,.01,100000,1e-4)
ψ(x)=(A_vmc/(pi))^(1/4)*exp(-A_vmc*x^2)
ψ²(x)=(A_vmc/(pi))^(1/2)*exp(-2A_vmc*x^2)
Ĥ(x)=1/2*(1/ψ(x)* -ψ''(x)+x^2)
X=metropolis_hastings(.5,0.0,1.0,n,ψ²)
X_new=blocking(X,10)
Ē_vmc,ΔĒ_vmc=jackknife(X_new,Ĥ)

## plot the results to compare 
plot(As,Es,yerr=Δs,label="numerical solution: delta_E")
E₀=round(min_e;digits=3)
A₀=round(min_A;digits=3)
E_vmc=round(Ē_vmc;digits=3)
A_vmc=round(A_vmc)

scatter!(As,Es,color="orange",label="numerical solution: min E_0=$E₀, min A_0=$A₀",legend=:bottomright,legendfontsize=5,markersize=3)
annotate!(.8, .75, text("epsilon: .5", :black, :right, 5))
annotate!(.8, .74, text("iterations: 30000", :black, :right, 5))
annotate!(.8, .73, text("number per bloc: 10", :black, :right, 5))
vline!([A₀],color="purple",label="numerical solution minimum: A_0=$A₀" ,line=[:dash])
vline!([A_vmc],color="green",label="numerical solution using gradient descent: A_vmc=$A_vmc" ,line=[:dash])
hline!([Ē],color="red",label="analytic solution: E_0=.5" ,line=[:dash])
hline!([E₀],color="purple",label="numerical solution minimum: E_0=$E₀" ,line=[:dash])
hline!([E_vmc],color="black",label="numerical solution minimum: E_vmc=$E_vmc , delta_E=$ΔĒ_vmc" ,line=[:dash])
title!("Calculating E_0 and delta_E wrt to paramter A: Minimizing E_0 wrt to A")
xlabel!("A")
ylabel!("<E> (h*omega)")

