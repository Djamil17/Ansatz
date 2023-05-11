"""

Description: 

defines the data structures necessary for arbitrary dimensional ansatz problem 
"""

struct Ω{T<:Real} ## position matrix , each row is a dimension 
    𝐫::SMatrix{T}
end 

struct Θ{T<:Real} ## 
    θ::Vector{T}
end 

struct QParams{T<:Real}
    n::Int ## number of spatial dimensions
    𝐫::{r{T}}
    Θ::{Θ{T}}
end 