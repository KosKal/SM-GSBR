function henon(θ,x,y)
  return(θ[1] + θ[2] * x + θ[3] * y + θ[4] * x * y + θ[5] * x ^ 2 + θ[6] * y ^ 2)
end


function noiseMix(p1::Float64, lam1::Float64, lam2::Float64)
  if rand() < p1
    z = rand(Normal(0, sqrt(1 / lam1)))
  else
    z = rand(Normal(0, sqrt(1 / lam2)))
  end
  return z
end

function saveDataNoise(n, seed, p1, lam1, lam2)
    srand(seed)
    x = zeros(n)
    for i in 1:1:n
        x[i] = noiseMix(p1, lam1, lam2)
    end
    savedir = 1000 + seed
    writedlm(string("/home/kkaloudis/Documents/Wishart Project/fq/Results/Sim 7/high/seed$savedir/datanoise.txt"), x)
end

function genData(n, Θ, x0, x00, p1, lam1, lam2, seed)
  srand(seed)
  x = zeros(n)

  x[1] = henon(Θ,x0,x00) + noiseMix(p1, lam1, lam2)
  x[2] = henon(Θ,x[1],x0) + noiseMix(p1, lam1, lam2)
  for i in 3:1:n
    x[i] = henon(Θ,x[i-1],x[i-2]) + noiseMix(p1, lam1, lam2)
  end

  return x
end

function noiseMixf1(lam::Float64)
  u = rand()
  z = 0.
  if u < 0.25
    z = rand(Normal(0,sqrt(1/lam)))
  elseif u < 0.5
    z = rand(Normal(0, sqrt(6*1/lam)))
  elseif u < 0.75
    z = rand(Normal(0,sqrt(11*1/lam)))
  else
    z = rand(Normal(0,sqrt(16*1/lam)))
  end
  return z
end

function genDataf1(n, Θ, x0, x00, lam, seed)
  srand(seed)
  x = zeros(n)

  x[1] = henon(Θ,x0,x00) + noiseMixf1(lam)
  x[2] = henon(Θ,x[1],x0) + noiseMixf1(lam)
  for i in 3:1:n
    x[i] = henon(Θ,x[i-1],x[i-2]) + noiseMixf1(lam)
  end

  return x
end

function unifmixrnd(x)
  local U::Float64
  n = length(x)
  nc = 0
  for i in 1:2:n
    nc += x[i + 1] - x[i]
  end

  U = 0
  u = rand()
  prob = 0
  for i in 1:2:n
    prob += (x[i + 1] - x[i]) / nc
    if u .< prob
      U = x[i] + (x[i + 1] - x[i]) * rand()
      break
    end
  end
  return(U)
end

function rangeIntersection(A::Array{Float64}, B::Array{Float64})
#=
Purpose: Range/interval intersection

 A and B two ranges of closed intervals written
 as vectors [lowerbound1 upperbound1 lowerbound2 upperbound2]
 or as matrix [lowerbound1, lowerbound2, lowerboundn;
               upperbound1, upperbound2, upperboundn]
 A and B have to be sorted in ascending order

 out is the mathematical intersection A n B

 EXAMPLE USAGE:
   >> out=rangeIntersection([1 3 5 9],[2 9])
   	out =  [2 3 5 9]
   >> out=rangeIntersection([40 44 55 58], [42 49 50 52])
   	out =  [42 44]
=#

# Allocate, as we don't know yet the size, we assume the largest case
  out1 = zeros(length(B)+(length(A)-2))
  k = 1

  while isempty(A) .== 0 && isempty(B) .== 0
  # make sure that first is ahead second
    if A[1] .> B[1]
      temp = copy(B)
      B = copy(A)
      A = copy(temp)
    end

    if A[2] .< B[1]
      A = copy(A[3:end])
      continue
    elseif A[2] .== B[1]
      out1[k] = B[1]
      out1[k + 1] = B[1]
      k = k + 2

      A = A[3:end]
      continue
    else
      if A[2] .== B[2]
        out1[k] = B[1]
        out1[k+1] = B[2]
        k = k + 2

        A = A[3:end]
        B = B[3:end]

      elseif A[2] .< B[2]
        out1[k] = B[1]
        out1[k+1] = A[2]
        k = k + 2

        A = A[3:end]
      else
        out1[k] = B[1]
        out1[k+1] = B[2]
        k = k + 2

        B = B[3:end]
      end
     end
    end

  # Remove the tails
  out = copy(out1[1:k-1])
  return(out)
end

function etgeornd(p::Float64, k::Vector{Int64})
local z::Vector{Int64}
  z = floor.(log.(rand(length(k))) ./ log.(1 - p)) + k
  return(z)
end

function cfy(yi, ypp, yp, yn, ynn, l1, l2, l3, theta)

  cc = -0.5 * ( l1 * (yi - henon(theta, yp, ypp)) ^ 2 + l2 * (yn - henon(theta, yi, yp)) ^ 2 + l3 * (ynn - henon(theta, yn, yi)) ^ 2 )

  return (cc)
end
