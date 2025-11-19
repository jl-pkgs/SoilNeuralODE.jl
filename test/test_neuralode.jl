using SoilNeuralODE
using Lux, Random, ComponentArrays
using RTableTools, DataFrames
using DifferentialEquations, SciMLSensitivity

# 加载数据
function load_data(f; n_train=240)
  df = fread(f)
  θ_obs = Matrix{Float32}(df[:, [:SOIL_MOISTURE_5, :SOIL_MOISTURE_10, :SOIL_MOISTURE_20,
                                   :SOIL_MOISTURE_50, :SOIL_MOISTURE_100]])'
  P = Float32.(df.P_CALC ./ 10.0)
  return θ_obs[:, 1:n_train], P[1:n_train]
end

# 参数设置
data_file = joinpath(@__DIR__, "..", "data", "SM_AR_Batesville_8_WNW_2024.csv")
θ_obs, P = load_data(data_file)
θ₀ = θ_obs[:, 1]

soil_params = (0.45f0, 0.05f0, 0.01f0, 0.02f0, 1.5f0)  # θ_s, θ_r, Ks, α, n
depths = Float32.([5, 10, 20, 50, 100])
Δz = 10.0f0
tspan = (0.0f0, Float32(size(θ_obs, 2) * 3600))

# 创建混合模型
nn_model = Chain(Dense(5 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 5))
rng = Xoshiro(123)
ps, st = Lux.setup(rng, nn_model)
ps = ComponentArray(ps)

hybrid_node = HybridNeuralODE(
  nn_model, st, tspan, depths, soil_params, Δz;
  alg=Tsit5(), reltol=1e-4, abstol=1e-6,
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  verbose=false
)

# 训练前评估
println("[训练前评估]")
gof_before = evaluate(hybrid_node, θ₀, ps, θ_obs)
println(gof_before)

# 训练
node, ps_trained, gof = train(hybrid_node, ps, θ₀, θ_obs; nepoch=50, step=10)

# 训练后评估
println("\n[训练后评估]")
println(gof)
