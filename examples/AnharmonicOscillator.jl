"""
Script: AnharmonicOscillator.jl
Author: Djamil Lakhdar-Hamina
Date: 05/04/2023
Description:

"""

include("../src/VmcProblem.jl")

const λ=.02f0
# const α=-.50f0
# const boundary=(-1.0,1.0)

particle_number=1
epoch=100
ϵ=1.50f0
samples=500
width=100
depth=3
output=1
η=.05

E₀_perturbation(λ)=1/2+3/4*λ-21/8*λ^2
ϕ(𝐱::Vector{T1}) where {T1<:AbstractFloat}=1/2*𝐱[1]^2+λ*𝐱[1]^4

function (m::NeuralAnsatz)(x)
    return exp.(m.chain(x).+α*(x[1].^2))
end

Flux.@functor NeuralAnsatz

# ansatz=buildNeuralNetAnsatz(particle_number,width,depth,output,celu)
# vmc_problem=vmcProblem(Ω=boundary,particle_number=particle_number,spatial_dimension=one_d,ϕ=ϕ)
# vmc_solution=vmcSolution(ψ=ansatz ,epoch=epoch,distribution=Uniform,ϵ=ϵ,N=samples,opt=Adam,η=η)
# @time E₀=runVmcProblem(vmc_problem,vmc_solution,Ē)

function test_width(widths)
    variational_info=[]
    for width in widths
        println("Calculating $width ...")
        ansatz=buildNeuralNetAnsatz(particle_number,width,depth,output,celu)
        vmc_problem=vmcProblem(Ω=boundary,particle_number=particle_number,spatial_dimension=one_d,ϕ=ϕ)
        vmc_solution=vmcSolution(ψ=ansatz ,epoch=epoch,distribution=Uniform,ϵ=ϵ,N=samples,opt=Adam,η=η)
        @time E₀,ΔE=runVmcProblem(vmc_problem,vmc_solution,jackknife,(vmc_solution.ψ,))
        push!(variational_info,(width,E₀,ΔE))
        println("Calculation finished!")
    end 
    return variational_info
end

function test_depth(depths)
    variational_info=[]
    for depth in depths
        println("Calculating $depth ...")
        ansatz=buildNeuralNetAnsatz(particle_number,width,depth,output,celu)
        vmc_problem=vmcProblem(Ω=boundary,particle_number=particle_number,spatial_dimension=one_d,ϕ=ϕ)
        vmc_solution=vmcSolution(ψ=ansatz ,epoch=epoch,distribution=Uniform,ϵ=ϵ,N=samples,opt=Adam,η=η)
        @time E₀,ΔE=runVmcProblem(vmc_problem,vmc_solution,jackknife,(vmc_solution.ψ,))
        push!(variational_info,(depth,E₀,ΔE))
        println("Calculation finished!")
    end 
    return variational_info
end

function main()
    return test_depth([1,2,3,4,5]),test_width([1,10,20,50,100,200])
end

variational_info=main()
depth_E,width_E=variational_info[1],variational_info[2]
E1,uncertainty1=[t[2] for t in depth_E],[t[3] for t in depth_E]
E2,uncertainty2=[t[2] for t in width_E],[t[3] for t in width_E]

plot([1,2,3,4,5],E1,yerr=uncertainty1)
title!("Depth vrs. Energy")
xlabel!("Depth")
ylabel!("Energy (a.u)")

plot([1,10,20,50,100,200],E2,yerr=uncertainty2)
title!("Iterations vrs. Energy")
xlabel!("Iteration")
ylabel!("Energy (a.u)")
