"""

""" 

using ForwardDiff

mag(𝐯::Vector)=sqrt(sum(𝐯.^2))

##TODO : multidimensional gradient descent (done)
##TODO: stochastic gradient descent 
##TODO: adaptive gradient descents  
##TODO: training set GD 

function gradient_descent(objective::Function, θ::Float64, η::Float64 ,iterations::Int64,tolerance::Float64)::Float64

    """

    gradient_descent(objective::Function, θ::Float64, η::Float64 ,iterations::Int64,tolerance::Float64)::Float64

    Performs gradient descent in n-d , for n variables, minimizes objective f with n arguments, given a guess vector θ and a learning rate η.
    Exits once the absolute value of the step η*||∇(f)|| < tolerance. 

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x)=x^2+2.324245x
    julia> gradient_descent(f,-1,.00001,1000000,1e-6)
    
    -1.1621220000061123


    """

    i=0
    xᵢ=θ
    while i<iterations
        loss=∇(objective,xᵢ)
        xᵢ₊₁= xᵢ-η*loss
        if abs(loss)<tolerance
            break
        end 
        xᵢ=xᵢ₊₁
        i+=1
    end 

    return xᵢ

end 


function gradient_descent(objective::Function, θ::Array{Float64,1}, η::Float64 ,iterations::Int64,tolerance::Float64)::Vector

    """

    gradient_descent(objective::Function, θ::Float64, η::Float64 ,iterations::Int64,tolerance::Float64)::Float64

    Performs gradient descent in n-d , for n variables, minimizes objective f with n arguments, given a guess vector θ and a learning rate η.
    Exits once the absolute value of the step η*||∇(f)|| < tolerance. 

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x::Vector)=x[1]^2+2.324245x[2]
    julia> gradient_descent(f,[-1.0,-2.0],.00001,1000000,1e-6)
    
    2-element Array{Float64,1}:
    -2.0607414274390493e-9
    -25.242449999476282
        

    """
    i=0
    wᵢ=θ
    
    while i<iterations
        loss=∇ᵣ(objective,wᵢ)
        wᵢ₊₁= wᵢ-η*loss
        if mag(loss)<tolerance
            break
        end 
        wᵢ=wᵢ₊₁
        i+=1
    end 

    return wᵢ

end 

function stochastic_gradient_descent(objective::Function, θ::Array{Float64,1}, η::Float64 ,iterations::Int64,tolerance::Float64)::Vector

    i=0
    wᵢ=θ
    
    while i<iterations
        sample=rand(wᵢ)
        loss=∇(objective,sample)
        wᵢ₊₁= wᵢ-η* loss
        if mag(loss)<tolerance
            break
        end 
        wᵢ=wᵢ₊₁
        i+=1
    end 

    return wᵢ

end 

function minibatch_sgd()

end 

function Δϵ(ψ::Function, Ĥ::Function, ε₀::Function, N::Integer,𝐱::Vector, Aᵢ) 
    𝒪=zeros(3) ## expected values 
    for i in 1:N
        log∇=∇(log_ψ,[𝐱[i], Aᵢ])[end]
        𝒪[1]+=1/ψ([𝐱[i], Aᵢ])*log∇*Ĥ([𝐱[i], Aᵢ],ψ)
        𝒪[2]+=ε₀([𝐱[i], Aᵢ],ψ)
        𝒪[3]+=log∇
    end 
    return 2*𝒪[1]/N-2*𝒪[2]/N*𝒪[3]/N
end 

function vmc(N,θ,η,iterations,tolerance,ψ²,Ĥ,ε₀)
    i=0
    Aᵢ=θ
    while i<iterations
        X=metropolis_hastings(ϵ,xmin,xmax,N,Aᵢ,ψ²)
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





