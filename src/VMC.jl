"""
Script:AnharmonicOscillator.jl
Author:Djamil Lakhdar-Hamina
Date: 05/04/2023
Description:

"""

using ForwardDiff
using StaticArrays 
using Plots
import LinearAlgebra:tr,Diagonal


## TODO: replace ansatz with neural network 
## TODO: defensive programming against mismatch in parameter dimension and spatial dimension and hamiltonian
## TODO: optimize VMC 
##          1. make sure gradient function efficient 
##          2. make sure creation and modifcation of mutable structs is efficient  
##          3. make sure calculation of gradient is efficient 
## TODO: fix bugs 
##      1. sometimes result of vmc return NaN 
##      2. sometimes certain wave functions need to be complex 
##      3. sometimes when gradient of energy is evaluated log is negative, so error thrown up
## TODO: write function annotations 

include("QState.jl")
include("MetropolisHastings.jl")
include("Jackknife.jl")

# include("GardientDescent.jl")
e=ℯ
xmin=0.0
xmax=2.0
ϵ=1.5 ## chosen because I observed acceptance rates, at this value acceptance is aroujnd 60%
n=1
m=1
tolerance=1e-6
η=.1
N=10000
iter=500

mag(𝐱::Vector)=sqrt(sum(𝐱.^2))
mag(𝐱::Number)=sqrt(sum(𝐱^2))

∇(g::Function,𝐱::Vector,m::Int,n::Int)=ForwardDiff.gradient(g,𝐱)[m:n]
∇²(g::Function,𝐱::Vector,m::Int,n::Int)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[m:n])
Δ(g::Function,𝐱::Vector,m::Int,n::Int)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[m:n])

anharmonic_oscillator(λ)=1/2+3/4*λ-21/8*λ^2

ψ(𝐱::Vector)=(𝐱[2]/π)^(1/4)*exp(-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2
ϕ(𝐱::Vector)=1/2*𝐱[1]^2+λ*𝐱[1]^4
Ĥ(𝐱::Vector, ψ::Function,m::Int,n::Int)=-∇²(ψ,𝐱,m,n)/2+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::Function,m::Int,n::Int)=ψ(𝐱)^-1*Ĥ(𝐱,ψ,m,n) ## ground state energy 

function Ē(𝐱::Matrix,θ::Vector, ψ::Function,m::Int,n::Int)
    N=length(𝐱)
    sum_=0
    for i in 1:N
        sum_+=ε₀([𝐱[i];θ],ψ,m,n)
    end 
    1/N*sum_
end 

function Δϵ(ψ::Function, Ĥ::Function, ε₀::Function, N::Integer,QState::MutQParams) 

    """

    Δϵ(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}

    """
    𝒪=zeros(3,QState.m) ## expected values 
    @simd for i in 1:N
        log∇=∇(𝐱->log(ψ(𝐱)),[QState.𝐫[i];QState.Θ],QState.n+1,QState.n+QState.m)
        @inbounds 𝒪[1,:]+=1/ψ([QState.𝐫[i];QState.Θ])*log∇*Ĥ([QState.𝐫[i];QState.Θ],ψ,1,QState.n)
        @inbounds 𝒪[2,1]+=ε₀([QState.𝐫[i];QState.Θ],ψ,1,QState.n)
        @inbounds 𝒪[3,:]+=log∇
    end 
    return 2*𝒪[1,:]/N-2*𝒪[2,1]/N*𝒪[3,:]/N
end 


function vmc(n::Number,m::Number,N::Number,η::Number,iterations::Int,tolerance::Number,ψ²::Function,Ĥ::Function,ε₀::Function)

    """

    vmc(N::Number,θ::QParams,η::NUmber,iterations::Int,tolerance::Number,ψ²::Function,Ĥ::Function,ε₀::Function)

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

    """

    if n<1 || n>3 
        ArgumentError("n, spatial dimensionality, must be between 1-3")
    end 

    i=1
    QState=MutQParams(n,m,zeros(N,n),rand(m)) ## randomnly initialize parameter vector
    while i<iterations
        QState.𝐫=metropolis_hastings(n,ϵ,xmin,xmax,N,QState.Θ,ψ²)
        loss=Δϵ(ψ,Ĥ,ε₀,N,QState)
        QState.Θ=QState.Θ.-η*loss
        if mag(loss)<tolerance
            break
        end 
        i+=1
    end 
    return QState
end 

function stochastic_vmc(n::Number,m::Number,N::Number,sample::Int,η::Number,iterations::Int,tolerance::Number,ψ²,Ĥ,ε₀)

    """

    vmc(N::Number,θ::QParams,η::NUmber,iterations::Int,tolerance::Number,ψ²::Function,Ĥ::Function,ε₀::Function)
    

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Real)=ℯ^-x^2
    julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

    """

    if n<1 || n>3 
        ArgumentError("n, spatial dimensionality, must be between 1-3")
    end 

    i=0
    QState=MutQParams(n,Ω(zeros(N,1)),Θ(rand(m)))
    while i<iterations
        QState.𝐫=metropolis_hastings(ϵ,xmin,xmax,sample,QState.Θ.θ,ψ²)
        loss=Δϵ(ψ,Ĥ,ε₀,N,QState)
        QState.θ=QState.θ-η*loss
        if abs(loss)<tolerance
            break
        end 
        i+=1
    end 
    return QState
end 


function run_test(λs,ψ²,Ĥ,ε₀,n,m)
    temp=Array{Tuple{Float64, Float64}}(undef, length(λs))
    for i in eachindex(λs) 
        global λ=λs[i]
        Qstate=vmc(n,m,N,η,iter,tolerance,ψ²,Ĥ,ε₀)
        temp[i]=jackknife(Qstate.𝐫[:,1],Qstate.Θ,ε₀,(ψ,1,1))
    end 
    return temp 
end 

Λ=collect(0:.002:.05)
analytic=anharmonic_oscillator.(Λ)

numeric_gaussian=run_test(Λ,ψ²,Ĥ,ε₀,1,1)
numeric_gaussian_1rst_order=run_test(Λ,𝐱->(((sqrt(pi)*(𝐱[3]+2*𝐱[2]))/(2*𝐱[2]^(3/2)))^(-1/2)*(1+𝐱[3]*𝐱[1]^2)*exp(-𝐱[2]*𝐱[1]^2))^2,Ĥ,ε₀,1,2)
numeric_gaussian_2nd_order=run_test(Λ,𝐱->(((sqrt(pi)*(3*𝐱[4]+2*𝐱[2]*𝐱[3]+4*𝐱[3]^2))/(4*𝐱[2]^(5/2)))^(-1/2)*(1+𝐱[3]*𝐱[1]^2+𝐱[4]*𝐱[1]^4)*exp(-𝐱[2]*𝐱[1]^2))^2,Ĥ,ε₀,1,3)

numeric_gaussian_energy=[n[1] for n in numeric_gaussian]
numeric_gaussian_error=[n[2] for n in numeric_gaussian]
numeric_gaussian_1rst_order_energy=[n[1] for n in numeric_gaussian_1rst_order]
numeric_gaussian_1rst_order_error=[n[2] for n in numeric_gaussian_1rst_order]
# numeric_gaussian_2nd_order_energy=[n[1] for n in numeric_gaussian_2nd_order]
# numeric_gaussian_2nd_order_error=[n[2] for n in numeric_gaussian_2nd_order]

plot(Λ,analytic,label=L"E_0(\lambda)=\frac{1}{2}+\frac{3}{4}\lambda-\frac{21}{8}\lambda^2",size=(1200,800))
plot!(Λ,numeric_gaussian_energy,yerr=numeric_gaussian_error,label=L"(\frac{A}{\pi})^\frac{1}{4}e^{-ax^2}")
plot!(Λ,numeric_gaussian_1rst_order_energy,yerr=numeric_gaussian_1rst_order_error,label=L"(\frac{2a^{\frac{3}{2}}}{(b+2a)\sqrt{\pi}})^{1/2}(1+bx^2)e^{-ax^2}")
# plot!(Λ,numeric_gaussian_2nd_order_energy,yerr=numeric_gaussian_2nd_order_error,label=L"(\frac{4a^\frac{5}{2}}{(3c+2ab+4a^2)\sqrt{\pi}})^{1/2}(1+bx^2+cx^4)e^{-ax^2}")

title!("Ground State Energy Anharmonic Oscillator: Perturbative vrs. Variational Results",titlefontsize=12)
xlabel!(L"\lambda")
ylabel!(L"E_0(\lambda) (a.u)")
annotate!(.04,.51,L"\eta=.01")
annotate!(.04,.508,L"iterations=10000")
annotate!(.04,.506,L"samples=10000")
