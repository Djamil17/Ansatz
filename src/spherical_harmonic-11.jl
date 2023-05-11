
using Plots

Y¹(θ)=3/(8π)*sin(θ)^2

θ=collect(0:2π:.01)

plot(θ,Y¹.(θ))
xlabel!("θ")
ylabel!("Y¹(θ)")
title!(L"$|Y(\theta)|$ vrs. $\theta$")
