
"""
Script: jackknife
Name: Djamil Lakhdar-Hamina
Date: 03/30/2023
Description: 

"""

## TODO: n dimensional jackknife method
## TODO: improve performance of jackknife (parallelize jackknife)
## TODO: annotations of functions
## TODO: enforce output types of functions
## TODO: test reduce versus loop solutions 

"""

"""
function blocking(X::Vector,number_per_block::Int)::Vector
    blocks=Iterators.partition(X, number_per_block)
    l=length(blocks)
    X_blocked=zeros(l)
    for (bloc,i) in zip(blocks,1:l)
        X_blocked[i]=sum(bloc)/number_per_block
    end
    return X_blocked
end 


"""

jackknife(experiment::Vector{AbstractFloat}, f::Function{AbstractFloat}):: where {T1<:AbstractFloat}

Calculates the expectation value and error for some function f of some experiment where f is a function and the results 
of the experiment are represented by an array of floats.

Examples
≡≡≡≡≡≡≡≡≡≡

julia> f(x)=x
julia> jackknife([1,2,3],f)

2,1


"""
function jackknife(experiment::Vector{T1}, f::Function)::Tuple{T1,T1} where {T1<:AbstractFloat}

    l=length(experiment)
    θᵢ=zeros(l)
    @fastmath @inbounds @simd for i in 1:l 
        θᵢ[i]=1/(l-1)*sum(f.([@view experiment[1:i-1];@view experiment[i+1:l]]))
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 

"""

jackknife(experiment::Array{Float64,1}, f::Function{Vector})

Calculates the expectation value and error for some function f of some experiment where f is a function and the results 
of the experiment are represented by an array of floats.

Examples
≡≡≡≡≡≡≡≡≡≡

julia> f(x)=x
julia> jackknife([1,2,3],f)

2,1


"""
function jackknife(experiment::Vector{T1},θ::Vector{T2},f::T3,extraargs::Tuple)  where {T1<:AbstractFloat,T2<:AbstractFloat,T3<:Function}
  

    l=length(experiment)
    θᵢ=zeros(l)
    @fastmath @inbounds @simd for i in 1:l 
        experiment_subset=[@view experiment[1:i-1];@view experiment[i+1:l]]
        sum_=0
        @fastmath @inbounds @simd for i in 1:l-1
            sum_+=f([experiment_subset[i];θ],extraargs...)
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 


"""

jackknife(experiment::Array{Float64,1}, f::Function{Vector})

Calculates the expectation value and error for some function f of some experiment where f is a function and the results 
of the experiment are represented by an array of floats.

Examples
≡≡≡≡≡≡≡≡≡≡

julia> f(x)=x
julia> jackknife([1,2,3],f)

2,1


"""
function jackknife(experiment::Vector{T1},θ::Vector{T2},f::T3,extraargs::Any) where {T1<:AbstractFloat,T2<:AbstractFloat,T3<:Function}

    l=length(experiment)
    θᵢ=zeros(l)
    @fastmath @inbounds @simd for i in 1:l 
        experiment_subset=[@view experiment[1:i-1];@view experiment[i+1:l]]
        sum_=0
        @fastmath @inbounds @simd for i in 1:l-1
            sum_+=f([experiment_subset[i];θ],extraargs)
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 


"""

jackknife(experiment::Array{Float64,1}, f::Function{Vector})

Calculates the expectation value and error for some function f of some experiment where f is a function and the results 
of the experiment are represented by an array of floats.

Examples
≡≡≡≡≡≡≡≡≡≡

julia> f(x)=x
julia> jackknife([1,2,3],f)

2,1


"""
function jackknife(experiment::Vector{T1},f::T2) where {T1<:AbstractFloat,T2<:Function}

    l=length(experiment)
    θᵢ=zeros(l)
    @fastmath @inbounds @simd for i in 1:l 
        experiment_subset=[experiment[1:i-1];experiment[i+1:l]]
        sum_=0
        @fastmath @inbounds @simd for i in 1:l-1
            sum_+=f([experiment_subset[i]])[1]
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 



ψ(𝐱)=exp(-1/2*𝐱[1]^2)
ϕ(𝐱::Vector{T}) where {T<:AbstractFloat}=1/2*𝐱[1]^2+λ*𝐱[1]^4
Ĥ(𝐱::Vector{T1}, ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=-∇²(ψ,𝐱)/2 .+ϕ(𝐱)*ψ(𝐱)
ε₀(𝐱::T1,ψ::T2) where {T1<:AbstractFloat,T2<:Union{NeuralAnsatz,Function}}=ψ(𝐱).^-1 .*Ĥ(𝐱,ψ)
