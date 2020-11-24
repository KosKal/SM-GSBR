function fqBackPredGSB(x::Array{Float64}, T::Int64, gibbsIter::Int64, burnIn::Int64, gEps1::Float64, gEps2::Float64, thLow::Float64,
  thUp::Float64, zpLow::Float64, zpUp::Float64, zLow::Float64, zUp::Float64, papr::Float64, pbpr::Float64, thin::Int64, samplerSeed::Int64)


	srand(samplerSeed)

  filename = "/Results"

  savelocation = string(pwd(), filename, "/seed$samplerSeed")
  mkpath(savelocation)

	# initialization
	n = length(x)
  nn = Int((gibbsIter - burnIn) / thin)

  if T .> 0 # back-pred
    x = append!(zeros(T), x)
    for i in 1:T
      x[i] = rand(Uniform(-1.,1.))
    end
    ss = n + T
    xbpred = zeros(nn, T)
  else # simple GSBR
    ss = n
  end


  theta = zeros(6) # polynomial coefficients vector
  sampledTheta = zeros(nn, 6) # matrix to store sample thetas

  x0s = zeros(nn, 2)
  sx01, sx02 = 0.5, 0.5

  N = collect(1:1:ss)
  d = collect(1:1:ss)
  noise = zeros(nn)
  clusters = zeros(nn)
  ps = zeros(nn)
  p = 0.5
	toler = 1e-06


	display("Starting MCMC ...")

	##########################################################################
	# MCMC
	##########################################################################

	for iter in 1:gibbsIter


      # Compute weights up to ss
      ##########################################################################

      M = maximum(N)
      w = zeros(M)

      for i in 1:M
        w[i] = p * (1 - p)^(i - 1)
      end

      ## Sample precisions
      ##########################################################################

      Lambdas = zeros(M)

      for j in 1:M
        counter = 0.0
        temp = 0.0
        if d[1] .== j
          counter = counter + 1.0
          temp = (x[1] - henon(theta, sx01, sx02)) ^ 2
        end
        if d[2] .== j
          counter = counter + 1.0
          temp += (x[2] - henon(theta, x[1], sx01)) ^ 2
        end
        for i in 3:ss
          if d[i] .== j
            counter = counter + 1.0
            temp = temp + (x[i] - henon(theta, x[i-1], x[i-2])) ^ 2
          end
        end
        shapel = gEps1 + 0.5 * counter
        ratel = gEps2 + 0.5 * temp
        # println("shape: $shapel")
        # println("rate: $ratel")
        Lambdas[j] = rand(Gamma(shapel, 1. / ratel))
      end


      ## Sample the indicator variables d(i) for i=1,...,ss
      ##########################################################################

      nc1 = 0.0
      prob1 = 0.0
      for j in 1:1:N[1]
        nc1 = nc1 + Lambdas[j]^0.5 * exp(-0.5 * Lambdas[j] * (x[1] - henon(theta, sx01, sx02))^2)
      end
      rd = rand()
      for j in 1:1:N[1]
        prob1 = prob1 + Lambdas[j]^0.5 * exp(-0.5 * Lambdas[j] * (x[1] - henon(theta, sx01, sx02))^2) / nc1
        if rd .< prob1
          d[1] = j
          break
        end
      end

      nc2 = 0.0
      prob2 = 0.0
      for j in 1:1:N[2]
        nc2 = nc2 + Lambdas[j]^0.5 * exp(-0.5 * Lambdas[j] * (x[2] - henon(theta, x[1], sx01))^2)
      end
      rd = rand()
      for j in 1:1:N[2]
        prob2 = prob2 + Lambdas[j]^0.5 * exp(-0.5 * Lambdas[j] * (x[2] - henon(theta, x[1], sx01))^2) / nc2
        if rd .< prob2
          d[2] = j
          break
        end
      end

      for i in 3:1:ss
        nc = 0.0
        prob = 0.0
        for j in 1:1:N[i]
          nc = nc + Lambdas[j]^0.5 * exp(-0.5 * Lambdas[j] * (x[i] - henon(theta, x[i-1], x[i-2]))^2)
        end
        rd = rand()
        for j in 1:1:N[i]
          prob = prob + Lambdas[j]^0.5 * exp(-0.5 * Lambdas[j] * (x[i] - henon(theta, x[i-1], x[i-2]))^2) / nc
          if rd .< prob
            d[i] = j
            break
          end
        end
      end

      # Sample N_i auxilliary variables
      ##########################################################################

      N = etgeornd(p, d)

      ## Number of clusters
      ##########################################################################

      ucl = length(unique(d))
      # println("clusters: $ucl")
      # @printf("uCl: %s \n", ucl)

      ## Sample geometricx probability
      ##########################################################################

      p = rand(Beta(papr + 2 * ss, pbpr + sum(N) - ss))


      ## Sample θ₀ = theta[1]
      ##########################################################################

      muth = Lambdas[d[1]] * (x[1] - (theta[5] * sx01^2 + theta[4] * sx01 * sx02 + theta[6] * sx02^2 + theta[2] * sx01 + theta[3] * sx02)) +
             Lambdas[d[2]] * (x[2] - (theta[5] * x[1]^2 + theta[4] * x[1] * sx01 + theta[6] * sx01^2 + theta[2] * x[1] + theta[3] * sx01))

      tauth = Lambdas[d[1]] + Lambdas[d[2]]

      for j = 3:ss
        muth += Lambdas[d[j]] * (x[j] - (theta[5] * x[j-1]^2 + theta[4] * x[j-1] * x[j-2] + theta[6] * x[j-2]^2 + theta[2] * x[j-1] + theta[3] * x[j-2]))
        tauth += Lambdas[d[j]]
      end
      muth = muth / tauth

      temp = -2. / tauth * log.(rand()) + (theta[1] - muth) ^ 2

      theta[1] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₁ = theta[2]
      # ##########################################################################
      #
      muth =  Lambdas[d[1]] * (x[1] * sx01 - (theta[5] * sx01^3 + theta[4] * sx01^2 * sx02 + theta[6] * sx01 * sx02^2 +
                              theta[3] * sx01 * sx02 + theta[1] * sx01)) +
              Lambdas[d[2]] * (x[2] * x[1] - (theta[5] * x[1]^3 + theta[4] * x[1]^2 * sx01 + theta[6] * x[1] * sx01^2 +
                              theta[3] * x[1] * sx01 + theta[1] * x[1]))

      tauth = Lambdas[d[1]] * sx01 ^ 2 + Lambdas[d[2]] * x[1] ^ 2

      for j = 3:ss
        muth += Lambdas[d[j]] * (x[j] * x[j-1] - (theta[5] * x[j-1]^3 + theta[4] * x[j-1]^2 * x[j-2] + theta[6] * x[j-1] * x[j-2]^2 +
                                theta[3] * x[j-1] * x[j-2] + theta[1] * x[j-1]))
        tauth += Lambdas[d[j]] * x[j - 1] ^ 2
      end
      muth = muth / tauth

      temp = -2. / tauth * log.(rand()) + (theta[2] - muth) ^ 2

      theta[2] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₂ = theta[3]
      # ##########################################################################
      #
      muth =  Lambdas[d[1]] * (x[1] * sx02 - (theta[5] * sx01^2 * sx02 + theta[4] * sx01 * sx02^2 + theta[6] * sx02^3 +
                              theta[2] * sx01 * sx02 + theta[1] * sx02)) +
              Lambdas[d[2]] * (x[2] * sx01 - (theta[5] * x[1]^2 * sx01 + theta[4] * x[1] * sx01^2 + theta[6] * sx01^3 +
                              theta[2] * x[1] * sx01 + theta[1] * sx01))

      tauth = Lambdas[d[1]] * sx02 ^ 2 + Lambdas[d[2]] * sx01 ^ 2

      for j = 3:ss
        muth += Lambdas[d[j]] * (x[j] * x[j-2] - (theta[5] * x[j-1]^2 * x[j-2] + theta[4] * x[j-1] * x[j-2]^2 + theta[6] * x[j-2]^3 +
                                theta[2] * x[j-1] * x[j-2] + theta[1] * x[j-2]))
        tauth += Lambdas[d[j]] * x[j - 2] ^ 2
      end

      muth = muth / tauth

      temp = -2. / tauth * log.(rand()) + (theta[3] - muth) ^ 2

      theta[3] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₃ = theta[4]
      # ##########################################################################
      #
      muth = Lambdas[d[1]] * (x[1] * sx01 * sx02 - (theta[5] * sx01^3 * sx02 + theta[6] * sx01 * sx02^3 + theta[2] * sx01^2 * sx02 +
                              theta[3] * sx01 * sx02^2 + theta[1] * sx01 * sx02)) +
             Lambdas[d[2]] * (x[2] * x[1] * sx01 - (theta[5] * x[1]^3 * sx01 + theta[6] * x[1] * sx01^3 + theta[2] * x[1]^2 * sx01 +
                              theta[3] * x[1] * sx01^2 + theta[1] * x[1] * sx01))

      tauth = Lambdas[d[1]] * sx01 ^ 2 * sx02 ^ 2 + Lambdas[d[2]] * x[1] ^ 2 *sx01 ^ 2

      for j = 3:ss
        muth += Lambdas[d[j]] * (x[j] * x[j-1] * x[j-2] - (theta[5] * x[j-1]^3 * x[j-2] + theta[6] * x[j-1] * x[j-2]^3 + theta[2] * x[j-1]^2 * x[j-2] +
                                theta[3] * x[j-1] * x[j-2]^2 + theta[1] * x[j-1] * x[j-2]))
        tauth += Lambdas[d[j]] * x[j - 1] ^ 2 * x[j - 2] ^ 2
      end

      muth = muth / tauth

      temp = -2. / tauth * log.(rand()) + (theta[4] - muth) ^ 2

      theta[4] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₄ = theta[5]
      # ##########################################################################
      #
      muth = Lambdas[d[1]] * (x[1] * sx01^2 - (theta[4] * sx01^3 * sx02 + theta[6] * sx01^2 * sx02^2 + theta[2] * sx01^3 +
                              theta[3] * sx01^2 * sx02 + theta[1] * sx01^2)) +
             Lambdas[d[2]] * (x[2] * x[1]^2 - (theta[4] * x[1]^3 * sx01 + theta[6] * x[1]^2 * sx01^2 + theta[2] * x[1]^3 +
                              theta[3] * x[1]^2 * sx01 + theta[1] * x[1]^2))

      tauth = Lambdas[d[1]] * sx01 ^ 4 + Lambdas[d[2]] * x[1] ^ 4

      for j = 3:ss
        muth += Lambdas[d[j]] * (x[j] * x[j-1]^2 - (theta[4] * x[j-1]^3 * x[j-2] + theta[6] * x[j-1]^2 * x[j-2]^2 + theta[2] * x[j-1]^3 +
                                theta[3] * x[j-1]^2 * x[j-2] + theta[1] * x[j-1]^2))
        tauth += Lambdas[d[j]] * x[j - 1] ^ 4
      end

      muth = muth / tauth

      temp = -2. / tauth * log.(rand()) + (theta[5] - muth) ^ 2

      theta[5] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₅ = theta[6]
      # ##########################################################################
      #
      muth = Lambdas[d[1]] * (x[1] * sx02^2 - (theta[5] * sx01^2 * sx02^2 + theta[4] * sx01 * sx02^3 + theta[2] * sx01 * sx02^2 +
                              theta[3] * sx02^3 + theta[1] * sx02^2)) +
             Lambdas[d[2]] * (x[2] * sx01^2 - (theta[5] * x[1]^2 * sx01^2 + theta[4] * x[1] * sx01^3 + theta[2] * x[1] * sx01^2 +
                              theta[3] * sx01^3 + theta[1] * sx01^2))

      tauth = Lambdas[d[1]] * sx02 ^ 4 + Lambdas[d[2]] * sx01 ^ 4

      for j = 3:ss
        muth += Lambdas[d[j]] * (x[j] * x[j-2]^2 - (theta[5] * x[j-1]^2 * x[j-2]^2 + theta[4] * x[j-1] * x[j-2]^3 + theta[2] * x[j-1] * x[j-2]^2 +
                                theta[3] * x[j-2]^3 + theta[1] * x[j-2]^2))
        tauth += Lambdas[d[j]] * x[j - 2] ^ 4
      end

      muth = muth / tauth

      temp = -2. / tauth * log.(rand()) + (theta[6] - muth) ^ 2

      theta[6] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))


      ## Sample past unobserved observations
      ##########################################################################

      if T .> 0

        if T .> 1
          for i in 1:1:(T-2)

            a4 = Lambdas[d[T-i+2]] * theta[5] ^ 2 + Lambdas[d[T-i+3]] * theta[6] ^ 2

            a3 = 2. * Lambdas[d[T-i+2]] * (theta[4] * theta[5] * x[T-i] + theta[2] * theta[5]) + 2. * Lambdas[d[T-i+3]] * (theta[4] * theta[6] * x[T-i+2] + theta[3] * theta[6])

            a2 = Lambdas[d[T-i+2]] * (theta[4]^2 * x[T-i]^2 + 2. * theta[5] * theta[6] * x[T-i]^2 + 2. * theta[2] * theta[4] * x[T-i] + 2. * theta[3] * theta[5] * x[T-i] -
                                2. * theta[5] * x[T-i+2] + 2. * theta[1] * theta[5] + theta[2]^2) +
               Lambdas[d[T-i+3]] * (theta[4]^2 * x[T-i+2]^2 + 2. * theta[5] * theta[6] * x[T-i+2]^2 + 2 * theta[2] * theta[6] * x[T-i+2] + 2 * theta[3] * theta[4] * x[T-i+2] -
                                2. * theta[6] * x[T-i+3] + 2. * theta[1] * theta[6] + theta[3]^2) +
                Lambdas[d[T-i+1]]

            a1 = 2. * (Lambdas[d[T-i+2]] * (theta[4] * theta[6] * x[T-i]^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * x[T-i]^2 + (-theta[4] * x[T-i+2] + theta[1] * theta[4] + theta[2] * theta[3]) * x[T-i] -
                                      theta[2] * x[T-i+2] + theta[1] * theta[2]) +
                    Lambdas[d[T-i+3]] * (theta[4] * theta[5] * x[T-i+2]^3 + (theta[2] * theta[4] + theta[3] * theta[5]) * x[T-i+2]^2 -
                                    theta[4] * x[T-i+2] * x[T-i+3] + (theta[1] * theta[4] + theta[2] * theta[3]) * x[T-i+2] - theta[3] * x[T-i+3] + theta[1] * theta[3]) - 
                    Lambdas[d[T-i+1]] * (theta[4] * x[T-i-1] * x[T-i] + theta[5] * x[T-i]^2 + theta[6] * x[T-i-1]^2 + theta[2] * x[T-i] + theta[3] * x[T-i-1] + theta[1])  )

            aux = -2. * log.(rand()) + a1 * x[T-i+1] + a2 * x[T-i+1] ^ 2 + a3 * x[T-i+1] ^ 3 + a4 * x[T-i+1] ^ 4

            poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
            allRoots = Polynomials.roots(Poly(poly))
            sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
            realRoots = sort(real(allRoots[sel]))
            intervals = rangeIntersection(realRoots, [zpLow; zpUp])

            x[T-i+1] = unifmixrnd(intervals) ##### go backwards ##### !!!!!!!!!!!!!!!!!!!!!!

          end

          # 2nd last unobserved past observation
          a4 = Lambdas[d[3]] * theta[5] ^ 2 + Lambdas[d[4]] * theta[6] ^ 2

          a3 = 2. * Lambdas[d[3]] * (theta[4] * theta[5] * x[1] + theta[2] * theta[5]) + 2. * Lambdas[d[4]] * (theta[4] * theta[6] * x[3] + theta[3] * theta[6])

          a2 = Lambdas[d[3]] * (theta[4]^2 * x[1]^2 + 2. * theta[5] * theta[6] * x[1]^2 + 2. * theta[2] * theta[4] * x[1] + 2. * theta[3] * theta[5] * x[1] -
                                2. * theta[5] * x[3] + 2. * theta[1] * theta[5] + theta[2]^2) +
               Lambdas[d[4]] * (theta[4]^2 * x[3]^2 + 2. * theta[5] * theta[6] * x[3]^2 + 2 * theta[2] * theta[6] * x[3] + 2 * theta[3] * theta[4] * x[3] -
                                2. * theta[6] * x[4] + 2. * theta[1] * theta[6] + theta[3]^2) +
               Lambdas[d[2]]

          a1 = 2. * (Lambdas[d[3]] * (theta[4] * theta[6] * x[1]^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * x[1]^2 + (-theta[4] * x[3] + theta[1] * theta[4] + theta[2] * theta[3]) * x[1] -
                                      theta[2] * x[3] + theta[1] * theta[2]) +
                    Lambdas[d[4]] * (theta[4] * theta[5] * x[3]^3 + (theta[2] * theta[4] + theta[3] * theta[5]) * x[3]^2 -
                                    theta[4] * x[3] * x[4] + (theta[1] * theta[4] + theta[2] * theta[3]) * x[3] - theta[3] * x[4] + theta[1] * theta[3]) - 
                    Lambdas[d[2]] * (theta[4] * x[1] * sx01 + theta[5] * x[1]^2 + theta[6] * sx01^2 + theta[2] * x[1] + theta[3] * sx01 + theta[1])  )

          aux = -2. * log.(rand()) + a1 * x[2] + a2 * x[2] ^ 2 + a3 * x[2] ^ 3 + a4 * x[2] ^ 4

          poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
          allRoots = Polynomials.roots(Poly(poly))
          sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
          realRoots = sort(real(allRoots[sel]))
          intervals = rangeIntersection(realRoots, [zpLow; zpUp])
          x[2] = unifmixrnd(intervals)

        end
        
        # Last unobserved past observation
        a4 = Lambdas[d[2]] * theta[5] ^ 2 + Lambdas[d[3]] * theta[6] ^ 2

        a3 = 2. * Lambdas[d[2]] * (theta[4] * theta[5] * sx01 + theta[2] * theta[5]) + 2. * Lambdas[d[3]] * (theta[4] * theta[6] * x[2] + theta[3] * theta[6])

        a2 = Lambdas[d[2]] * (theta[4]^2 * sx01^2 + 2. * theta[5] * theta[6] * sx01^2 + 2. * theta[2] * theta[4] * sx01 + 2. * theta[3] * theta[5] * sx01 -
                              2. * theta[5] * x[2] + 2. * theta[1] * theta[5] + theta[2]^2) +
             Lambdas[d[3]] * (theta[4]^2 * x[2]^2 + 2. * theta[5] * theta[6] * x[2]^2 + 2 * theta[2] * theta[6] * x[2] + 2 * theta[3] * theta[4] * x[2] -
                              2. * theta[6] * x[3] + 2. * theta[1] * theta[6] + theta[3]^2) +
             Lambdas[d[1]]

        a1 = 2. * (Lambdas[d[2]] * (theta[4] * theta[6] * sx01^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * sx01^2 + (-theta[4] * x[2] + theta[1] * theta[4] + theta[2] * theta[3]) * sx01 -
                                    theta[2] * x[2] + theta[1] * theta[2]) +
                  Lambdas[d[3]] * (theta[4] * theta[5] * x[2]^3 + (theta[2] * theta[4] + theta[3] * theta[5]) * x[2]^2 -
                                  theta[4] * x[2] * x[3] + (theta[1] * theta[4] + theta[2] * theta[3]) * x[2] - theta[3] * x[3] + theta[1] * theta[3]) - 
                  Lambdas[d[1]] * (theta[4] * sx01 * sx02 + theta[5] * sx01^2 + theta[6] * sx02^2 + theta[2] * sx01 + theta[3] * sx02 + theta[1])  )

        aux = -2. * log.(rand()) + a1 * x[1] + a2 * x[1] ^ 2 + a3 * x[1] ^ 3 + a4 * x[1] ^ 4

        poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
        allRoots = Polynomials.roots(Poly(poly))
        sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
        realRoots = sort(real(allRoots[sel]))
        intervals = rangeIntersection(realRoots, [zpLow; zpUp])
        x[1] = unifmixrnd(intervals)

      end



      ## Sample initial conditions x₀,y₀ = x0,x00
      ##########################################################################

      # x₀

      a4 = Lambdas[d[1]] * theta[5] ^ 2 + Lambdas[d[2]] * theta[6] ^ 2

      a3 = 2. * Lambdas[d[1]] * (theta[4] * theta[5] * sx02 + theta[2] * theta[5]) + 2. * Lambdas[d[2]] * (theta[4] * theta[6] * x[1] + theta[3] * theta[6])

      a2 = Lambdas[d[1]] * (theta[4]^2 * sx02^2 + 2. * theta[5] * theta[6] * sx02^2 + 2. * theta[2] * theta[4] * sx02 + 2. * theta[3] * theta[5] * sx02 -
                            2. * theta[5] * x[1] + 2. * theta[1] * theta[5] + theta[2]^2) +
           Lambdas[d[2]] * (theta[4]^2 * x[1]^2 + 2. * theta[5] * theta[6] * x[1]^2 + 2 * theta[2] * theta[6] * x[1] + 2 * theta[3] * theta[4] * x[1] -
                            2. * theta[6] * x[2] + 2. * theta[1] * theta[6] + theta[3]^2)

      a1 = 2. * (Lambdas[d[1]] * (theta[4] * theta[6] * sx02^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * sx02^2 + (-theta[4] * x[1] + theta[1] * theta[4] + theta[2] * theta[3]) * sx02 -
                                  theta[2] * x[1] + theta[1] * theta[2]) +
                Lambdas[d[2]] * (theta[4] * theta[5] * x[1]^3 + (theta[2] * theta[4] + theta[3] * theta[5]) * x[1]^2 -
                                theta[4] * x[1] * x[2] + (theta[1] * theta[4] + theta[2] * theta[3]) * x[1] - theta[3] * x[2] + theta[1] * theta[3]))

      aux = -2. * log.(rand()) + a1 * sx01 + a2 * sx01 ^ 2 + a3 * sx01 ^ 3 + a4 * sx01 ^ 4

      poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
      #allRoots = roots(poly, polish=true, epsilon=1e-20)
      allRoots = Polynomials.roots(Poly(poly))

      #println("allRoots = $allRoots")
      # sel = imag(allRoots) .== 0.0
      # realRoots = sort(real(allRoots[sel]))

      sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
      realRoots = sort(real(allRoots[sel]))
      intervals = rangeIntersection(realRoots, [zLow; zUp])
      sx01 = unifmixrnd(intervals)
      #
      # x00 = y₀

      a4  = Lambdas[d[1]] * theta[6] ^ 2

      a3 = 2. * theta[6] * Lambdas[d[1]] * (theta[4] * sx01 + theta[3])

      a2 = Lambdas[d[1]] * ( -2. * theta[6] * (x[1] - theta[1] - theta[2] * sx01 - theta[5] * sx01^2) + (theta[4] * sx01 + theta[3])^2 )

      a1 = -2. * Lambdas[d[1]] * (x[1] - theta[1] - theta[5] * sx01^2 - theta[2] * sx01) * (theta[4] * sx01 + theta[3])

      aux = -2. * log.(rand()) + a1 * sx02 + a2 * sx02 ^ 2 + a3 * sx02 ^ 3 + a4 * sx02 ^ 4

      poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
      allRoots = Polynomials.roots(Poly(poly))
      sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
      # sel = imag(allRoots) .== 0.0 # treat as real the roots with imagine part < ϵ
      realRoots = sort(real(allRoots[sel]))
      intervals = rangeIntersection(realRoots, [zLow; zUp])
      sx02 = unifmixrnd(intervals)


      ## After Burn-In period
      ###############################wwwwwwwww###########################################

      if (iter .> burnIn) & ((iter-burnIn) % thin .== 0)

        ii = Int((iter - burnIn)/thin)

        ## Sample noise predictive
        ##########################################################################

        cW = cumsum(w)
        flag = rand()
        if cW[end] .< flag
          pred = rand(Normal(0, sqrt(1 / rand(Gamma(gEps1, 1 / gEps2))))) # draw from the prior
        else
          for j in 1:1:length(cW)
            if flag .< cW[j]
               pred = rand(Normal(0, sqrt(1 / Lambdas[j])))
               break
            end
          end
        end

        # Store values
        sampledTheta[ii, :] = theta
        x0s[ii, :] = [sx01 sx02]
        noise[ii] = pred
        clusters[ii] = ucl
        ps[ii] = p

        if T .> 0
          xbpred[ii,:] = x[1:1:T]
        end


      end

      if iter % 50000 .== 0
         println("MCMC Iterations: $iter")
      end

	end

	display("... MCMC finished !")

	## Write values in .txt files - specific path
	##########################################################################

  if T .> 0 
    writedlm(string(savelocation,"/xbpred.txt"), xbpred)
  end 
	writedlm(string(savelocation, "/thetas.txt"), sampledTheta)
	writedlm(string(savelocation,"/x0s.txt"), x0s)
  writedlm(string(savelocation,"/noise.txt"), noise, '\n')
	writedlm(string(savelocation,"/ucl.txt"), clusters, '\n')
  writedlm(string(savelocation,"/ps.txt"), ps, '\n')

 # #, xbpred
 return x0s
 
end
