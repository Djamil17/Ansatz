include("MetropolisHastings.jl")
include("Jackknife.jl")
include("GardientDescent.jl")

## testing 

# test on arbitrary S(x)
# ϵ=.1
# N=100000
# X=metropolis_hastings(ϵ,xmin,xmax,N,S)
# y=ℯ.^-S.(X)
# Z=quadgk(f, -Inf, Inf, rtol=1e-3)[1]
# Ē=1/N*sum(X.^2)
# histogram(X,normed=true,label="binned values")
# scatter!(X,y./Z,label="pi*e^-S(x)")
# vline!([Ē],color="red",label="<x^2>")
# title!("Sampling of pi(x)ℯ^(x²-x⁴) : ϵ=.1 , N=100000")
# xlabel!(L"x")
# ylabel!(L"pi(x) ℯ^-S(x)")

## test on gaussian
# ϵ=.5
# gaussian(x)=x^2
# g(x)=exp(-x^2)
# X=metropolis_hastings(ϵ,xmin,xmax,N,gaussian)
# y=g.(X)
# Z=quadgk(g, -Inf, Inf, rtol=1e-3)[1]
# Ē=sum(1/N*X.^2)
# histogram(X,normed=true)
# scatter!(X,y./Z)
# vline!([Ē],color="red")
# title!("Sampling of pi(x)ℯ^(x²-x⁴) : ϵ=.1 , N=100000")
# xlabel!(L"x")
# ylabel!(L"pi(x) ℯ^-S(x)")

## test on multidimensional gaussian in x,y,z coordinates  
# gaussian_3d(x,y,z)=x^2+y^2+z^2
# ## test on maxwell boltzmann distribution χ² distribution
# ħ=1.054e-34
# m=1
# A=.7
# ω=.7ħ
# N=1000000

## find 2-d min
z(𝐫::Vector)=-exp(-((𝐫[1])^2+(𝐫[2])^2))
gradient_descent(z,[-0.0,-2.0],.0001,1000,1e-5 )

## find weights for a linear model 

"""
x1  x2  y
4	1	2
2	8	-14
1	0	1
3	2	-1
1	4	-7
6	7	-8
"""

J(w::Vector)=1/12*(w[1]*x1+w[2]*x2+w[3]-y)

w0=[1.2342,2.334,3]
x1=[4,2,1,3,1,6]
x2=[2,-14,1,-1,-7,-8]
y=[2,-14,1,-1,-7,-8]

∇ᵣ.(J,w0)


ForwardDiff.gradient(x -> ForwardDiff.gradient(x->g(x[1],x[2]),[1.0,1.0]),
[1.0,1.0])