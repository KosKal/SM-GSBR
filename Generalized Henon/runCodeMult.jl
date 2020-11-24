cd("INSERT_PATH")

using Distributions
using Polynomials
using DataFrames

include("toolBox.jl")
include("ghmBGSBR.jl")

#########################################################
# Parameter Declaration
#########################################################

# map parameters
theta, x0, y0 = [0., 2. ,-0.1, 0.3], 0.5, 1.

# sample size
ss = 1000
# past prediction horizon 
T = 2

# noise f_{2,l} precisions
w1, lam1 = 0.9, 1e07
lam2 = 1e02

# corresponding deterministic orbit
xdet = zeros(ss+T)
xdet[1] = ghm(theta, x0, y0)
xdet[2] = ghm(theta, xdet[1], x0)
for i in 3:(ss+T)
  xdet[i] = ghm(theta, xdet[i-1], xdet[i-2])
end

dataSeed = 29
# fulldata = copy(xdet) # no noise
# fulldata = genDataf1(ss + T, theta, x0, y0, lam2, dataSeed) #f_{1} noise
fulldata = genData(ss + T, theta, x0, y0, w1, lam1, lam2, dataSeed) #f_{2,l} noise
predValues = fulldata[1:T]
data = fulldata[(T+1):end]


# GSBR reconstruction - bPrediction
gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zLow, zUp, papr, pbpr, thin, samplerSeed =
 150000, 50000, 3., 0.3, -10.0, 10.0, -5.0, 5.0, 1., 10., 10, 2105;

filename = "/Results"
savelocation = string(pwd(), filename, "/seed$samplerSeed")
mkpath(savelocation)
writedlm(string(savelocation,"/xdet.txt"), xdet, '\n')
writedlm(string(savelocation,"/predValues.txt"), predValues)
writedlm(string(savelocation,"/data.txt"), data, '\n')

@time f = open("PATH/smGHM.txt", "a") do f
	for i in 1:150
		println("#######################################")
	    println("Simulation: $i/150")
		println("#######################################")
		x0s = ghmBGSBR(data[280+i:end], T, gibbsIter, burnIn, gEps1, gEps2, thLow, thUp, zLow, zUp, papr, pbpr, thin, samplerSeed);
		writedlm(f, x0s)
		end
	x0s = 0
	gc()
end


