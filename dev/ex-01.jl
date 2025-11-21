using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, 
  Optimization, OptimizationOptimJL,
  OptimizationOptimisers, Random, Plots, Enzyme

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
ode_data_vec = [ode_data[:, i] for i in 1:size(ode_data, 2)]

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat=tsteps)

function predict_neuralode(p)
  prob_neuralode(u0, p, st)[1].u
end

function loss_neuralode(p)
  pred = predict_neuralode(p)
  loss = 0.0f0
  for i in 1:length(pred)
    diff = ode_data_vec[i] .- pred[i]
    loss += sum(abs2, diff)
  end
  return loss
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
function callback(state, l; doplot=false)
  println(l)
  # plot current prediction against data
  if doplot
    pred = predict_neuralode(state.u)
    pred_mat = reduce(hcat, pred)
    plt = scatter(tsteps, ode_data[1, :]; label="data")
    scatter!(plt, tsteps, pred_mat[1, :]; label="prediction")
    display(plot(plt))
  end
  return false
end

pinit = ComponentArray(p)
callback((; u=pinit), loss_neuralode(pinit); doplot=false)  # 关闭绘图

# use Optimization.jl to solve the problem
# 测试 Enzyme 自动微分后端
Enzyme.API.strictAliasing!(false)
adtype = Optimization.AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
  optprob, OptimizationOptimisers.Adam(0.05); callback=callback, maxiters=300)

# optprob2 = remake(optprob; u0=result_neuralode.u)
# result_neuralode2 = Optimization.solve(
#   optprob2, Optim.BFGS(; initial_stepnorm=0.01); callback, allow_f_increases=false)
# callback((; u=result_neuralode2.u), loss_neuralode(result_neuralode2.u); doplot=true)
