"""
Script: Colagero-Moser.jl 
Name: Djamil Lakhdar-Hamina
Date: 06/30/2023
Description: 

"""

include("../src/VmcProblem.jl")

const g=1.0f00
const a=1.0f00
const boundary=(-.50f0,.50f0)

Ω=1f-3
particle_number=2
ϵ=1.50f0
samples=100
width=100
depth=1
output=1
η=.0001f0
α=-0.5f0 
ω=1

Eₖ(k,N)=ω*(k+N/2+N*(N-1)/2*(1/2*(1+2*g^2)^(1/2)+1/2))

function buildϕ(particleN::T1,interparticle_force::String="sinh") where {T1<:Integer}

    header="g^2*a^2*("
    closing=")"
    middleString=""
    for i in 1:particleN, j in 1:particleN
        if i!=j && i>j 
            if i==particleN && j==particleN-1
            middleString*="$interparticle_force.(a*(x[$i]-x[$j])).^-2"
            else 
            middleString*="$interparticle_force.(a*(x[$i]-x[$j])).^-2 + "
            end 
        end 
    end 

    chainString=header*middleString*closing
    return @eval(x -> $(Meta.parse(chainString)))
end

function NewtonGirard(v)  
    degrees=length(v)
    return [[x^degree for degree in 1:degrees] for x in v] |> sum
end 

# ϕ=buildϕ(2)

ϕ(𝐱::Vector{T1}) where {T1<:AbstractFloat}= .05f0*ω*(𝐱[1].^2 + 𝐱[2].^2)+g^2*(𝐱[1]-𝐱[2]).^-2

function (m::NeuralAnsatz)(x)
        
    return exp.(Ω* m.chain(sort(x)).+α*(x[1].^2 .+x[2].^2))
end

Flux.@functor NeuralAnsatz

function main()
    info=[]
    for epoch in [1,10,100,200,500]
        ansatz=buildNeuralNetAnsatz(particle_number,width,depth,output,relu)
        vmc_problem=vmcProblem(Ω=boundary,particle_number=particle_number,spatial_dimension=one_d,ϕ=ϕ)
        vmc_solution=vmcSolution(ψ=ansatz ,epoch=epoch,distribution=Uniform,ϵ=ϵ,N=samples,opt=Adam,η=η)
        @time E₀=runVmcProblem(vmc_problem,vmc_solution,Ē,(vmc_solution.ψ,))
        push!(info,(E₀))
    end
    return info
end 

@show Eₖ(0,2)
info=main()


