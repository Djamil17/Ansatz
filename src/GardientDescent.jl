"""

""" 

using ForwardDiff

mag(𝐯::Vector)=sqrt(sum(𝐯.^2))

∇=ForwardDiff.derivative
∇=ForwardDiff.gradient

##TODO : multidimensional gradient descent (done)
##TOD): stochastic gradient descent 
##TOD: adaptive gradient descents  

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
        xᵢ₊₁= xᵢ-η* loss
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
        loss=∇(objective,wᵢ)
        wᵢ₊₁= wᵢ-η* loss
        if mag(loss)<tolerance
            break
        end 
        wᵢ=wᵢ₊₁
        i+=1
    end 

    return wᵢ

end 


z(𝐫::Vector)=-exp(-((𝐫[1])^2+(𝐫[2])^2))

gradient_descent(z,[-0.0,-2.0],.0001,1000000,1e-5 )