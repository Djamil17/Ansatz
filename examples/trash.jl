"""
Script:AnharmonicOscillator.jl
Author:Djamil Lakhdar-Hamina
Date: 05/04/2023
Description:

"""

using ForwardDiff, StaticArrays 
import LinearAlgebra:tr,Diagonal

include("MetropolisHastings.jl")
include("Jackknife.jl")
# include("QState.jl")
# include("GardientDescent.jl")

xmin=0.0
xmax=2.0
ϵ=1.5 ## chosen because I observed acceptance rates, at this value acceptance is aroujnd 60%
λ=0.0

∇(g::Function,𝐱::Vector)=ForwardDiff.gradient(g,𝐱)
∇²(g::Function,𝐱::Vector)=sum(Diagonal(ForwardDiff.hessian(g,𝐱)))
Δ(g::Function,𝐱::Vector)=sum(Diagonal(ForwardDiff.hessian(g,𝐱)))

∇ᵣ(g::Function,𝐱::Vector)=Diagonal(ForwardDiff.gradient(g,𝐱))[1:end-1]
∇²ᵣ(g::Function,𝐱::Vector)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[1:end-1])
Δᵣ(g::Function,𝐱::Vector)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[1:end-1])

# ∇θ(g::Function,𝐱)=ForwardDiff.gradient(g,𝐱)
# ∇θ(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)
# Δθ(g::Function,𝐱::Vector)=ForwardDiff.hessian(g,𝐱)

ψ(𝐱::Vector)=(𝐱[2]/π)^(1/4)*exp(-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2
log_ψ(𝐱::Vector)=log(ψ(𝐱))
ϕ(𝐱::Vector)=1/2*𝐱[1]^2+λ*𝐱[1]^4
Ĥ(𝐱::Vector, ψ::Function)=-∇²ᵣ(ψ,𝐱)/2+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::Function)=ψ(𝐱)^-1*Ĥ(𝐱,ψ) ## ground state energy 

function Δϵ(ψ::Function, Ĥ::Function, ε₀::Function, N::Integer,𝐱::Vector, Aᵢ::Number) 

    """

    Δϵ(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    
    

    """

    𝒪=zeros(3) ## expected values 
    for i in 1:N
        log∇=∇(log_ψ,[𝐱[i], Aᵢ])[end]
        𝒪[1]+=1/ψ([𝐱[i], Aᵢ])*log∇*Ĥ([𝐱[i], Aᵢ],ψ)
        𝒪[2]+=ε₀([𝐱[i], Aᵢ],ψ)
        𝒪[3]+=log∇
    end 
    return 2*𝒪[1]/N-2*𝒪[2]/N*𝒪[3]/N
end 

# function Δϵ(ψ::Function, Ĥ::Function, ε₀::Function, N::Integer,Aᵢ) 

#     """

#     Δϵ(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}

#     """

#     𝒪=zeros(3) ## expected values 
#     for i in 1:N
#         log∇=∇(log_ψ,[𝐱[i], Aᵢ])[end]
#         @show log∇
#         𝒪[1]+=1/ψ([𝐱[i], Aᵢ])*log∇*Ĥ([𝐱[i], Aᵢ],ψ)
#         𝒪[2]+=ε₀([𝐱[i], Aᵢ],ψ)
#         𝒪[3]+=log∇
#     end 
#     return 2*𝒪[1]/N-2*𝒪[2]/N*𝒪[3]/N
# end 

# function vmc(N::Number,θ::Number,η::Number,iterations::Int,tolerance::Number,ψ²,Ĥ,ε₀)

#     """

#     vmc(ϵ::Float64,xmin::Float64,xmax::Float64,iterations::Int64,P::Function)::Array{Float64}
    
    

#     """

#     i=0
#     Aᵢ=θ
#     while i<iterations
#         X=metropolis_hastings(ϵ,xmin,xmax,N,Aᵢ,ψ²)
#         loss=Δϵ(ψ,Ĥ,ε₀, N,X,Aᵢ)
#         # @show loss
#         Aᵢ₊₁=Aᵢ-η*loss
#         if abs(loss)<tolerance
#             break
#         end 
#         Aᵢ=Aᵢ₊₁
#         i+=1
#     end 
#     return Aᵢ
# end 

function vmc(N::Number,θ::Number,η::Number,iterations::Int,tolerance::Number,ψ²,Ĥ,ε₀)

    """

    vmc(N::Number,θ::QParams,η::NUmber,iterations::Int,tolerance::Number,ψ²::Function,Ĥ::Function,ε₀::Function)
    

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> 
    julia> 
    

    """

    i=0
    Aᵢ=θ
    while i<iterations
        X=metropolis_hastings(ϵ,xmin,xmax,N,θ,ψ²)
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


## test 

## exact anharmonic as function of lambda 
anharmonic_oscillator(λ)=1/2+3/4*λ-21/8*λ^2

## numeric using one-parameter gaussian ansatz 

Λ=collect(0:.1:1)
analytic=zeros(length(Λ))
numeric=zeros(length(Λ))
for i in 1:length(Λ) 
    analytic[i]=anharmonic_oscillator(Λ[i])
    global λ=Λ[i]
    numeric[i]=vmc(10000,.4,.01,100000,1e-8,ψ²,Ĥ,ε₀)
end 

plot(Λ,analytic)
plot!(Λ,numeric)


"""
Script:AnharmonicOscillator.jl
Author:Djamil Lakhdar-Hamina
Date: 05/04/2023
Description:

"""

using ForwardDiff
using StaticArrays 
using Plots
using LaTeXStrings
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

include("../src/QState.jl")
include("../src/MetropolisHastings.jl")
include("../src/Jackknife.jl")

# include("GardientDescent.jl")
e=ℯ
xmin=0.0
xmax=2.0
ϵ=1.5 ## chosen because I observed acceptance rates, at this value acceptance is aroujnd 60%
n=1
m=2
tolerance=1e-6
const λ=0.0
η=.1
N=1000
iter=500

mag(𝐱::Vector)=sqrt(sum(𝐱.^2))
mag(𝐱::Number)=sqrt(sum(𝐱^2))

rel(x)=x < 0.0 ? 0.0 :  x  
σ(x)=rel(x)

∇(g::Function,𝐱::Vector,m::Int,n::Int)=ForwardDiff.gradient(g,𝐱)[m:n]
∇²(g::Function,𝐱::Vector,m::Int,n::Int)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[m:n])
Δ(g::Function,𝐱::Vector,m::Int,n::Int)=sum(Diagonal(ForwardDiff.hessian(g,𝐱))[m:n])

ψ(𝐱::Vector)=exp(σ(𝐱[2]*𝐱[1]+𝐱[3]*𝐱[1])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2
ϕ(𝐱::Vector)=1/2*𝐱[1]^2+λ*𝐱[1]^4
Ĥ(𝐱::Vector, ψ::Function,m::Int,n::Int)=-∇²(ψ,𝐱,m,n)/2+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::Vector,ψ::Function,m::Int,n::Int)=ψ(𝐱)^-1*Ĥ(𝐱,ψ,m,n) ## ground state energy 

function Ē(𝐱::Matrix,θ::Vector, ψ::Function,m::Int,n::Int)
    N=length(𝐱)
    sum_=0
    for i in 1:N
        sum_+=ε₀([𝐱[i]])
    end 
    1/N*sum_
end 

## gradient of energy 
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

# function stochastic_vmc(n::Number,m::Number,N::Number,sample::Int,η::Number,iterations::Int,tolerance::Number,ψ²,Ĥ,ε₀)

#     """

#     vmc(N::Number,θ::QParams,η::NUmber,iterations::Int,tolerance::Number,ψ²::Function,Ĥ::Function,ε₀::Function)
    

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x::Real)=ℯ^-x^2
#     julia> metropolis_hastings(.1, -2.0,-1.0,10000,f)
    

#     """

#     if n<1 || n>3 
#         ArgumentError("n, spatial dimensionality, must be between 1-3")
#     end 

#     i=0
#     QState=MutQParams(n,Ω(zeros(N,1)),Θ(rand(m)))
#     while i<iterations
#         QState.𝐫=metropolis_hastings(ϵ,xmin,xmax,sample,QState.Θ.θ,ψ²)
#         loss=Δϵ(ψ,Ĥ,ε₀,N,QState)
#         QState.θ=QState.θ-η*loss
#         if abs(loss)<tolerance
#             break
#         end 
#         i+=1
#     end 
#     return QState
# end 

# function run_test(λs,ψ²,Ĥ,ε₀,n,m)
#     temp=Array{Tuple{Float64, Float64}}(undef, length(λs))
#     for i in eachindex(λs) 
#         global λ=λs[i]
#         Qstate=vmc(n,m,N,η,iter,tolerance,ψ²,Ĥ,ε₀)
#         temp[i]=jackknife(Qstate.𝐫[:,1],Qstate.Θ,ε₀,(ψ,1,1))
#     end 
#     return temp 
# end 

# Λ=collect(0:.002:.05)
# analytic=anharmonic_oscillator.(Λ)


## with multiple nodes 
rs=[]
E=[]
ψs=[]

ground_state=[1/sqrt(2)*(1/pi)^(1/4)*exp(-(x/2)^2) for x in collect(-3.5:.1:3.5)]

ψ(𝐱::Vector)=exp(σ(𝐱[2]*𝐱[1]+𝐱[3])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2

qstate=vmc(n,m,N,η,iter,1e-4,ψ²,Ĥ,ε₀)
push!(rs,sort(qstate.𝐫[:,1]))
push!(E,jackknife(qstate.𝐫[:,1],qstate.Θ,ε₀,(ψ,1,1)))
push!(ψs,[ψ([rs[1][i];qstate.Θ]) for i in 1:N ])

ψ(𝐱::Vector)=exp(σ(𝐱[2]*𝐱[1]+𝐱[3]+𝐱[4]*𝐱[1]+𝐱[5])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2

qstate=vmc(n,4,N,η,iter,1e-4,ψ²,Ĥ,ε₀)
push!(rs,sort(qstate.𝐫[:,1]))
push!(E,jackknife(qstate.𝐫[:,1],qstate.Θ,ε₀,(ψ,1,1)))
push!(ψs,[ψ([rs[2][i];qstate.Θ]) for i in 1:N ])

ψ(𝐱::Vector)=exp(σ(𝐱[2]*𝐱[1]+𝐱[3]+𝐱[4]*𝐱[1]+𝐱[5]+𝐱[6]*𝐱[1]+𝐱[5])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2

qstate=vmc(n,6,N,η,iter,1e-4,ψ²,Ĥ,ε₀)
push!(rs,sort(qstate.𝐫[:,1]))
push!(E,jackknife(qstate.𝐫[:,1],qstate.Θ,ε₀,(ψ,1,1)))
push!(ψs,[ψ([rs[3][i];qstate.Θ]) for i in 1:N ])


plot(collect(-3.5:.1:3.5),ground_state,label=L"Analytic:(\frac{2}{\pi})^{1/4}e^{x/2^2} ",size=(1000,500))

plot!(rs[1],ψs[1]./(2*maximum(ψs[1])),label="One-Node, One-Layer",size=(1000,500))
plot!(rs[2],ψs[2]./(2*maximum(ψs[2])),label="Two-Node, One-Layer")
plot!(rs[3],ψs[3]./(2*maximum(ψs[3])),label="Three-Node, One-Layer")
title!("Ansätze Eigenstates of Harmonic Oscillator: Single-Layer Multi-Node Normalized Neural Networks")
xlabel!(L"\bf{x}(a.u.)")
ylabel!(L"\psi(\bf{x})")
annotate!(-1.5,8,L"iterations=500")
annotate!(-1.5,7.5,L"samples=1000")
annotate!(-1.5, 7.0, L"\eta=.1")


## with multiple layers 
ψ(𝐱::Vector)=exp(σ(𝐱[2]*𝐱[1]+𝐱[3])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2

qstate=vmc(n,m,N,η,iter,1e-4,ψ²,Ĥ,ε₀)
push!(rs,sort(qstate.𝐫[:,1]))
push!(E,jackknife(qstate.𝐫[:,1],qstate.Θ,ε₀,(ψ,1,1)))
push!(ψs,[ψ([rs[4][i];qstate.Θ]) for i in 1:N ])

ψ(𝐱::Vector)=exp(σ(σ(𝐱[2]*𝐱[1]+𝐱[3])+𝐱[4])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2

qstate=vmc(n,3,N,η,iter,1e-4,ψ²,Ĥ,ε₀)
push!(rs,sort(qstate.𝐫[:,1]))
push!(E,jackknife(qstate.𝐫[:,1],qstate.Θ,ε₀,(ψ,1,1)))
push!(ψs,[ψ([rs[5][i];qstate.Θ]) for i in 1:N ])

ψ(𝐱::Vector)=exp(σ(σ(σ(𝐱[2]*𝐱[1]+𝐱[3])+𝐱[4])+𝐱[5])-𝐱[2]*𝐱[1]^2)
ψ²(𝐱::Vector)=ψ(𝐱)^2

qstate=vmc(n,4,N,η,iter,1e-4,ψ²,Ĥ,ε₀)
push!(rs,sort(qstate.𝐫[:,1]))
push!(E,jackknife(qstate.𝐫[:,1],qstate.Θ,ε₀,(ψ,1,1)))
push!(ψs,[ψ([rs[6][i];qstate.Θ]) for i in 1:N ])

plot(collect(-3.5:.1:3.5),ground_state,label=L"Analytic:(\frac{2}{\pi})^{1/4}e^{x/2^2} ",size=(1000,500))

# plot!(rs[4],ψs[4]./(2*maximum(ψs[4])),label="One-Node, One-Layer",size=(1000,500))
# plot!(rs[5],ψs[5]./(2*maximum(ψs[5])),label="One-Node, Two-Layer")
# plot!(rs[6],ψs[6]./(2*maximum(ψs[6])),label="One-Node, Three-Layer")
# title!("Ansätze Eigenstates of Harmonic Oscillator: Single-Node Multi-Layer Normalized Neural Networks")
# xlabel!(L"\bf{x}(a.u.)")
# ylabel!(L"\psi(\bf{x})")
# annotate!(-1.5,8,L"iterations=500")
# annotate!(-1.5,7.5,L"samples=1000")
# annotate!(-1.5, 7.0, L"\eta=.1")

