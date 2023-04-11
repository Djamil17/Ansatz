using Random
using Distributions

struct Network2
    sizes::Array{Int,1}
    num_layers::Array{Int,1}
    biases::Array{Float64,undef}
    weights::Array{Float64,undef}
end 

sizes=[1,2,3]

biases=[rand(y, 1) for y in sizes[1:end]]
weights=[rand(y, x) for (x,y) in zip(sizes[], sizes[])]


function sigmoid(z)
    return 1.0/(1.0+exp(-z))

function feedforward(a,net::Network1)
    return sigmoid.(a.*net.weights+net.biases)

