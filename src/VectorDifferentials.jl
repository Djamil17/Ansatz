"""
Script: 

Description:

Short hands for various differential operations in vector calculus 

"""

∇(g::Function,𝐱::Vector)=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::Function,𝐱::Vector)=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))
∇(g::NeuralAnsatz,𝐱::Vector)=gradient(𝐱->sum(g(𝐱)),𝐱)
∇²(g::NeuralAnsatz,𝐱::Vector)=sum(Diagonal(hessian(𝐱->sum(g(𝐱)),𝐱)))

