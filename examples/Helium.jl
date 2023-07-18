"""

"""



function (m::NeuralAnsatz)(x)
    return exp.(symmetrizer(m.chain(x)))*ψ
end 

Flux.@functor NeuralAnsatz
chain=buildChain(particle_num,width,depth,σ)
Ψ=NeuralAnsatz(chain)
