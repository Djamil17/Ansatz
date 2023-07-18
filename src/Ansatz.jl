module 

    using Random,Distributions,Zygote,Flux

    include("Jackknife.jl")
    include("MetropolisHastings.jl")
    include("NeuralAnsatz.jl")
    include("NeuralAnsatzFuncs.jl")
    include("VectorDifferentials.jl")
    include("VmcProblem.jl")

end