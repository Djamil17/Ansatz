
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
# include("GardientDescent.jl")

xmin=0.0
xmax=2.0
ϵ=1.5 ## chosen because I observed acceptance rates, at this value acceptance is aroujnd 60%

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
        X=metropolis_hastings(ϵ,0.0,1.0,N,ψ²)
        Ē0=1/N*sum(Ĥ.(X))
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
        loss=2*( 1/N*sum(  ((-4Aᵢ.*X.^2 .-1)/4Aᵢ) .* (-(Aᵢ^5/π)^(1/4).*(2Aᵢ*X.^2 .-1)+ 1/2*X.^2)  )  - (1/N*sum(Ĥ.(X))*1/N*sum((-4Aᵢ*X.^2 .-1)/4Aᵢ) )   )
        Aᵢ₊₁=Aᵢ-η*loss
        if abs(loss)<tolerance
            break
        end 
        Aᵢ=Aᵢ₊₁
        i+=1
    end 
    return Aᵢ
end 

## compute stupid descent and good descent
# Ē=.5

# As=collect(.1:.1:2)
# n=100000
# Es, Δs=stupid_descent(n,As,50)
# min_e=minimum(Es)
# min_A=As[argmin(Es)]

# println(minimum(Δs))
# println(min_e)
# println(min_A)

# A_vmc=vmc_harmonic(50000,.2,.01,100000,1e-4)

# A=.500
# ψ(x)=(A/(π))^(1/4)*exp(-A*x^2)
# ψ²(x)=(A/(π))^(1/2)*exp(-2A*x^2)
# Ĥ(X::Real)=1/2*(1/ψ(X)* -ψ''(X)+X^2)

# function Ĥ(X::Vector)::Vector
#     l=length(X)
#     result=zeros(l)
#     for i in 1:l
#         result[i]=1.0/2.0*(1/ψ(X[i])* -ψ''(X[i])+X[i]^2)
#     end 
#     return result
# end 

# function Ĥ1(X::Vector)::Vector
#     l=length(X)
#     result=zeros(l)
#     for i in 1:l
#         @show X[i]
#         @show 1.0/2.0*(-2*A*(2*A*X[i]^2-1)+X[i]^2)
#         result[i]=1.0/2.0*(-2*A*(2*A*X[i]^2-1)+X[i]^2)
#     end 
#     return result
# end 

n=20000
X=metropolis_hastings(.6,0.0,1.0,n,ψ²)
X=X[Int(n/2)+1:end]
X_new=blocking(X,20)
Ē_vmc,ΔĒ_vmc=jackknife(X_new,Ĥ)
@show Ē_vmc,ΔĒ_vmc

## parametrize the wave function in terems of a neural network 


# ## plot the results to compare 
# plot(As,Es,yerr=Δs,label="numerical solution: delta_E")
# E₀=round(min_e;digits=3)
# A₀=round(min_A;digits=3)
# E_vmc=round(Ē_vmc;digits=3)
# A_vmc=round(A_vmc)

# scatter!(As,Es,color="orange",label="numerical solution: min E_0=$E₀, min A_0=$A₀",legend=:bottomright,legendfontsize=5,markersize=3)
# annotate!(.8, .75, text("epsilon: .5", :black, :right, 5))
# annotate!(.8, .74, text("iterations: 30000", :black, :right, 5))
# annotate!(.8, .73, text("number per bloc: 10", :black, :right, 5))
# vline!([A₀],color="purple",label="numerical solution minimum: A_0=$A₀" ,line=[:dash])
# vline!([A_vmc],color="green",label="numerical solution using gradient descent: A_vmc=$A_vmc" ,line=[:dash])
# hline!([Ē],color="red",label="analytic solution: E_0=.5" ,line=[:dash])
# hline!([E₀],color="purple",label="numerical solution minimum: E_0=$E₀" ,line=[:dash])
# hline!([E_vmc],color="black",label="numerical solution minimum: E_vmc=$E_vmc , delta_E=$ΔĒ_vmc" ,line=[:dash])
# title!("Calculating E_0 and delta_E wrt to paramter A: Minimizing E_0 wrt to A")
# xlabel!("A")
# ylabel!("<E> (h*omega)")

