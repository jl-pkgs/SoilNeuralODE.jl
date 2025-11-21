using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq,
  Optimization, OptimizationOptimJL,
  OptimizationOptimisers, Random

rng = Xoshiro(0)
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2]; length=datasize)

function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(); saveat=tsteps))

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat=tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
  pred = predict_neuralode(p)
  loss = sum(abs2, ode_data .- pred)
  return loss
end

function callback(state, l)
  println("Loss = $l")
  return false
end

pinit = ComponentArray(p)
println("Initial loss: ", loss_neuralode(pinit))

# 测试 Zygote（对比基准）
println("\n=== Testing Zygote ===")
@time adtype_zygote = Optimization.AutoZygote()
@time optf_zygote = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype_zygote)
optprob_zygote = Optimization.OptimizationProblem(optf_zygote, pinit)

println("\nRunning optimization with Zygote...")
@time result_zygote = Optimization.solve(
  optprob_zygote, OptimizationOptimisers.Adam(0.05); callback=callback, maxiters=20)

println("\nFinal loss (Zygote): ", loss_neuralode(result_zygote.u))
