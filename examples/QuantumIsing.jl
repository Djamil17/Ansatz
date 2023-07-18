using LinearAlgebra:kron


const σᶻ=[ 1+0im 0+0im ; 0+0im -1+0im]
const σʸ=[ 0+0im 0+-im ; 0+im 0+0im]
const σˣ=[ 0+0im 1+0im ; 1+0im 0+0im]

⊗(A,B) = kron(A,B)
sum(x,y)=∑(x,y)

- σᶻ⊗σᶻ - σᶻ⊗σᶻ - σᶻ⊗σᶻ-(σˣ + σˣ + σˣ + σˣ)

function TransverseFieldIsing(;N,h)
    id = [1 0; 0 1]
    σˣ = [0 1; 1 0]
    σᶻ = [1 0; 0 -1]
    
    # vector of operators: [σᶻ, σᶻ, id, ...]
    first_term_ops = fill(id, N)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ
    
    # vector of operators: [σˣ, id, ...]
    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ
    
    H = zeros(Int, 2^N, 2^N)
    for i in 1:N-1
        # tensor multiply all operators
        H -= foldl(⊗, first_term_ops)
        # cyclic shift the operators
        first_term_ops = circshift(first_term_ops,1)
    end
    
    for i in 1:N
        H -= h*foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops,1)
    end
    H
end