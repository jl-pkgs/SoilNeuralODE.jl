using SoilNeuralODE
using Lux, Random, ComponentArrays
using RTableTools, DataFrames
using DifferentialEquations, SciMLSensitivity
using Interpolations


# 加载数据
function load_data(f; n_train=240)
  df = fread(f)
  θ_obs = Matrix{Float32}(df[:, [:SOIL_MOISTURE_5, :SOIL_MOISTURE_10, :SOIL_MOISTURE_20,
    :SOIL_MOISTURE_50, :SOIL_MOISTURE_100]])'
  P = Float32.(df.P_CALC ./ 10.0)  # mm/h转为cm/h
  return θ_obs[:, 1:n_train], P[1:n_train]
end


# 参数设置
data_file = joinpath(@__DIR__, "..", "data", "SM_AR_Batesville_8_WNW_2024.csv")
θ_obs, P = load_data(data_file)
θ₀ = θ_obs[:, 1]

# 土壤参数（单位：cm/h）
θ_s = 0.40f0
θ_r = 0.08f0
Ks_surface = 3.6f0  # cm/h (=0.001 cm/s * 3600)
α = 0.01f0  # 1/cm
n = 1.3f0
depths = Float32.([5, 10, 20, 50, 100])  # cm

# 创建深度变化的土壤参数剖面
soil_params_profile = create_soil_profile(depths, θ_s, θ_r, Ks_surface, α, n; L_decay=80.0f0)

# 打印每层Ks值
println("Soil hydraulic conductivity profile [cm/h]:")
for (i, sp) in enumerate(soil_params_profile)
  println("  Depth $(depths[i])cm: Ks = $(sp[3]) cm/h")
end

# 创建降水插值函数（时间单位：小时）
tspan = (0.0f0, Float32(length(P)))  # 时间范围：小时
times = range(0.0f0, tspan[2], length=length(P) + 1)[1:end-1]
P_flux = P  # P已经是 cm/h 单位
P_interp = LinearInterpolation(times, P_flux, extrapolation_bc=Flat())


# 创建神经网络学习入渗修正（输入：P, θ_surface, K_surface = 3维）
nn_model = Chain(
  Dense(3 => 16, tanh),
  Dense(16 => 16, tanh),
  Dense(16 => 1))  # 输出：入渗修正项
rng = Xoshiro(123)
ps, st = Lux.setup(rng, nn_model)
ps = ComponentArray(ps)


# 使用Rosenbrock23，它对autodiff要求更宽松
hybrid_node = HybridNeuralODE(
  nn_model, st, tspan, depths, soil_params_profile, P_interp;
  alg=Rosenbrock23(autodiff=false), reltol=1e-3, abstol=1e-5,
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  verbose=false
)

# 训练前评估
println("\n[训练前评估]")
gof_before = evaluate(hybrid_node, θ₀, ps, θ_obs)
println(gof_before)

# 训练
node, ps_trained, gof = train(hybrid_node, ps, θ₀, θ_obs; nepoch=20, step=1)

# 训练后评估
println("\n[训练后评估]")
println(gof)
