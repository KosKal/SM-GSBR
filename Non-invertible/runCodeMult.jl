cd("INSERT_PATH")

using Distributions
using Polynomials
using DataFrames

include("toolBox.jl")
include("fqBackPredGSB.jl")

#########################################################
# Parameter Declaration
#########################################################

# map parameters
theta, x0, y0 = [1.38, 0., 0., 0., -1., 0.211], 1.5, 0.5

# sample size
ss = 250
# past prediction horizon 
T = 2

# noise f_{2,l} precisions
w1, lam1 = 0.9, 1e07
lam2 = 1e02

xdet = zeros(ss+T)
xdet[1] = henon(theta, x0, y0)
xdet[2] = henon(theta, xdet[1], x0)
for i in 3:(ss+T)
  xdet[i] = henon(theta, xdet[i-1], xdet[i-2])
end

dataSeed = 50
fulldata = copy(xdet) # no noise
# fulldata = genDataf1(ss + T, theta, x0, y0, lam2, dataSeed) #f_{1} noise
# fulldata = genData(ss + T, theta, x0, y0, w1, lam1, lam2, dataSeed) #f_{2,l} noise
predValues = fulldata[1:T]
data = fulldata[(T+1):end]


# GSBR reconstruction - bPrediction
gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zpLow, zpUp, zLow, zUp, papr, pbpr, thin, samplerSeed =
 50000, 25000, 1e-03, 1e-03, -10.0, 10.0, -2.0, 2.0, -10.0, 10.0, 10., 10., 5, 12345;

filename = "/Results"
savelocation = string(pwd(), filename, "/seed$samplerSeed")
mkpath(savelocation)
writedlm(string(savelocation,"/predValues.txt"), predValues)
writedlm(string(savelocation,"/data.txt"), data, '\n')

@time f = open("PATH/noninv.txt", "a") do f
	for i in 1:25
		println("#######################################")
	    println("Simulation: $i/25")
		println("#######################################")
		x0s = fqBackPredGSB(data[i:end], T, gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zpLow, zpUp, zLow, zUp, papr, pbpr, thin, samplerSeed);
		writedlm(f, x0s)
		end
	x0s = 0
	gc()
end

