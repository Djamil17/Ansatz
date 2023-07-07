
"""
Script: jackknife
Name: Djamil Lakhdar-Hamina
Date: 03/30/2023
Description: 

"""

##TODO: somehow need to enforce the sort of function to be inputed 

function blocking(X::Vector,number_per_block::Int)::Vector

    """

    """

    
    blocks=Iterators.partition(X, number_per_block)
    l=length(blocks)
    X_blocked=zeros(l)
    for (bloc,i) in zip(blocks,1:l)
        X_blocked[i]=sum(bloc)/number_per_block
    end
    return X_blocked
end 

## TODO: n dimensional jackknife method
## TODO: improve performance of jackknife (parallelize jackknife)

# function stupid_jackknife(experiment::Function,f::Function,S::Function,Ns::Array)

#     """

#     jackknife(experiment::Array{Float64,1}, f)::(Float64,Float64)

#     Calculates the expectation value and error for some function f of some experiment where f is a function and the results 
#     of the experiment are represented by an array of floats.

#     Examples
#     ≡≡≡≡≡≡≡≡≡≡

#     julia> f(x)=x
#     julia> jackknife([1,2,3],f)
    
#     2,1

#     """

#     l=length(Ns)
#     experiments=zeros(length(Ns))
#     θ=zeros(length(Ns))

#     for n in 1:l
#         experiments[n]=1/Ns[n]*sum(f.(experiment(ϵ,xmin,xmax,Ns[n],S)))
#     end 

#     jack_matrix=zeros((l,l))

#     for n in 1:l
#         for m in 1:l
#             if n==m
#                 jack_matrix[n,m]=0
#             else 
#             jack_matrix[m,n]=experiments[n]
#             end 
#         end 
#     end 

#     for row in 1:l
#         θ[row]=sum(jack_matrix[row,:])/(l-1)
#     end 

#     θ̄=sum(experiments)/l
#     Δθ=sqrt((l-1)/(l)*sum((θ.-θ̄).^2))

#     return  θ̄,Δθ
# end 

# f input returns a scalar
function jackknife(experiment::Array{Real,1}, f::Function)
    """

    jackknife(experiment::Array{Float64,1}, f::Function{Float64})

    Calculates the expectation value and error for some function f of some experiment where f is a function and the results 
    of the experiment are represented by an array of floats.

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> f(x)=x
    julia> jackknife([1,2,3],f)
    
    2,1


    """

    l=length(experiment)
    θᵢ=zeros(l)
    for i in 1:l 
        θᵢ[i]=1/(l-1)*sum(f.([experiment[1:i-1];experiment[i+1:l]]))
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 

## jackknife for vmc, when the experiment is parametrized not just by experiment varaiable but others 
function jackknife(experiment::Vector{T1},θ::Vector{T2},f::T3,extraargs::Tuple)  where {T1<:AbstractFloat,T2<:AbstractFloat,T3<:Function}
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

    l=length(experiment)
    θᵢ=zeros(l)
    @inbounds for i in 1:l 
        experiment_subset=[experiment[1:i-1];experiment[i+1:l]]
        sum_=0
        @inbounds for i in 1:l-1
            sum_+=f([experiment_subset[i];θ],extraargs...)
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 

function jackknife(experiment::Vector{T1},θ::Vector{T2},f::T3,extraargs::Any) where {T1<:AbstractFloat,T2<:AbstractFloat,T3<:Function}
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

    l=length(experiment)
    θᵢ=zeros(l)
    @inbounds for i in 1:l 
        experiment_subset=[experiment[1:i-1];experiment[i+1:l]]
        sum_=0
        @inbounds for i in 1:l-1
            sum_+=f([experiment_subset[i];θ],extraargs)
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 

function jackknife(experiment::Vector,f::T1) where {T1<:Function}
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

    l=length(experiment)
    θᵢ=zeros(l)
    @inbounds for i in 1:l 
        experiment_subset=[experiment[1:i-1];experiment[i+1:l]]
        sum_=0
        @inbounds for i in 1:l-1
            sum_+=f([experiment_subset[i]])[1]
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end 

function jackknife(experiment::Vector{T1},f::T2,extraargs::Any) where {T1<:AbstractFloat,T2<:Function}
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

    l=length(experiment)
    θᵢ=zeros(l)
    @inbounds for i in 1:l 
        experiment_subset=[experiment[1:i-1];experiment[i+1:l]]
        sum_=0
        @inbounds for i in 1:l-1
            sum_+=f([experiment_subset[i]],extraargs)[1]
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end


function jackknife(experiment::Vector{Vector{T1}},f::T2,extraargs::Any) where {T1<:AbstractFloat,T2<:Function}
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

    l=length(experiment)
    θᵢ=zeros(l)
    @inbounds for i in 1:l 
        experiment_subset=[experiment[1:i-1];experiment[i+1:l]]
        sum_=0
        @inbounds for i in 1:l-1
            sum_+=f([experiment_subset[i]],extraargs...)[1]
        end 
        θᵢ[i]=1/(l-1)*sum_
    end 
    θ̄=sum(θᵢ)/l
    Δθ=sqrt((l-1)/(l)*sum((θᵢ.-θ̄).^2))
    return θ̄,Δθ
end