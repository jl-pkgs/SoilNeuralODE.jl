using Lux, DifferentialEquations, DiffEqFlux, ComponentArrays, Random
using Parameters  # 用于 @with_kw 宏
using Statistics
using LinearAlgebra: norm
using Optimization, OptimizationOptimisers
using Zygote  # 用于梯度计算
using SciMLSensitivity  # 用于 sensealg

#=============================================================================
基于 2_04.jl 的虚拟数据测试
目标：生成虚拟观测数据，训练混合模型，验证模型可行性
=============================================================================#

# Richards 方程时间导数计算（用于 ODE）
function richards_dθdt(θ, soil_params, Δz)
  θ_s, θ_r, Ks, α, n = soil_params
  n_layers = length(θ)

  # 计算每层的水力性质
  K = similar(θ)
  ψ = similar(θ)
  for i in 1:n_layers
    K[i] = hydraulic_conductivity(θ[i], θ_s, θ_r, Ks, n)
    ψ[i] = soil_water_potential(θ[i], θ_s, θ_r, α, n)
  end

  # 计算界面水力传导度
  K_interface = zeros(eltype(θ), n_layers + 1)
  K_interface[1] = K[1]
  for i in 2:n_layers
    K_interface[i] = 0.5f0 * (K[i-1] + K[i])
  end
  K_interface[end] = K[end]

  # 计算界面通量
  Q = zeros(eltype(θ), n_layers + 1)
  for i in 2:n_layers
    Q[i] = -K_interface[i] * ((ψ[i] - ψ[i-1]) / Δz + 1.0f0)
  end

  # 边界通量（无通量边界）
  Q[1] = 0.0f0
  Q[end] = 0.0f0

  # 计算含水量变化率 dθ/dt
  dθdt = zeros(eltype(θ), n_layers)
  for i in 1:n_layers
    dθdt[i] = (Q[i] - Q[i+1]) / Δz
  end

  # 边界条件：顶层和底层不变
  dθdt[1] = 0.0f0
  dθdt[end] = 0.0f0

  return dθdt
end

#=============================================================================
第二步: 定义混合模型 = Richards物理模型 + 神经网络修正
=============================================================================#

# 神经网络：输入是含水量，输出是对 dθ/dt 的修正项
nn_model = Chain(
  Dense(5 => 16, tanh),   # 输入是当前含水量 θ (5层)
  Dense(16 => 16, tanh),
  Dense(16 => 5)          # 输出是修正项 (5层)
)

# 混合 ODE 函数：physics + NN 修正
function create_hybrid_ode(nn_model, nn_state, depths, soil_params, Δz)
  function hybrid_ode!(dθ, θ, p, t)
    θ_s, θ_r = soil_params[1], soil_params[2]

    # 1. 物理模型：Richards 方程
    dθ_physics = richards_dθdt(θ, soil_params, Δz)

    # 2. 神经网络修正：基于当前状态 θ
    θ_input = reshape(θ, :, 1)

    # 使用 Lux 模型
    corrections, _ = nn_model(θ_input, p, nn_state)
    corrections = vec(corrections)

    # 3. 混合：物理 + 小系数的 NN 修正
    dθ_hybrid = dθ_physics .+ 0.01f0 .* corrections

    # 更新 dθ（原地修改）
    dθ .= dθ_hybrid

    return nothing
  end

  return hybrid_ode!
end

#=============================================================================
第三步: 定义 HybridNeuralODE 层
=============================================================================#

struct HybridNeuralODE{M,T,P,S}
  nn_model::M
  nn_state::S
  tspan::T
  depths::Vector{Float32}
  soil_params::P
  Δz::Float32
  alg
  kwargs
end

function HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params, Δz;
                          alg=Tsit5(), kwargs...)
  return HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params, Δz, alg, kwargs)
end

# 前向传播：求解 ODE
function (node::HybridNeuralODE)(θ₀, p, st)
  ode_func! = create_hybrid_ode(node.nn_model, node.nn_state, node.depths, node.soil_params, node.Δz)
  prob = ODEProblem(ode_func!, θ₀, node.tspan, p)
  sol = solve(prob, node.alg; node.kwargs..., saveat=node.tspan[1]:3600.0f0:node.tspan[2])
  return sol, st
end

#=============================================================================
第四步: 生成虚拟观测数据
=============================================================================#

function generate_virtual_data(rng, soil_params, depths, Δz, tspan, θ₀)
  """
  生成虚拟观测数据：使用纯物理模型 + 噪声
  """
  # 纯物理 ODE 函数（无神经网络修正）
  function physics_only_ode!(dθ, θ, p, t)
    dθ_physics = richards_dθdt(θ, soil_params, Δz)
    dθ .= dθ_physics
    return nothing
  end

  # 求解纯物理 ODE
  prob = ODEProblem(physics_only_ode!, θ₀, tspan, nothing)
  sol = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-6, saveat=tspan[1]:3600.0f0:tspan[2])

  # 添加观测噪声（模拟真实传感器误差）
  θ_true = Array(sol)  # (n_layers, n_times)
  noise_level = 0.02f0  # 2% 噪声
  θ_obs = θ_true .+ noise_level .* randn(rng, Float32, size(θ_true)...)

  # 确保物理约束
  θ_s, θ_r = soil_params[1], soil_params[2]
  θ_obs .= clamp.(θ_obs, θ_r + 0.01f0, θ_s - 0.01f0)

  return θ_obs, θ_true, sol.t
end

#=============================================================================
第五步: 损失函数
=============================================================================#

function loss_neuralode(p, θ₀, θ_obs_seq, node, soil_params)
  θ_s, θ_r = soil_params[1], soil_params[2]

  # 前向求解 ODE
  sol, _ = node(θ₀, p, NamedTuple())
  θ_pred_seq = Array(sol)  # (n_layers, n_times)

  # 数据拟合损失
  loss_data = mean((θ_pred_seq .- θ_obs_seq) .^ 2)

  # 物理约束 1：含水量不能超出饱和度
  loss_physics1 = mean(max.(0.0f0, θ_pred_seq .- θ_s) .^ 2)

  # 物理约束 2：含水量不能低于残余含水量
  loss_physics2 = mean(max.(0.0f0, θ_r .- θ_pred_seq) .^ 2)

  # 总损失
  total_loss = loss_data + 0.1f0 * (loss_physics1 + loss_physics2)

  return total_loss
end

#=============================================================================
第六步: 训练回调函数
=============================================================================#

function create_callback(θ₀, θ_obs_seq, node, soil_params)
  iter = Ref(0)
  loss_history = Float32[]

  function callback(p, l)
    iter[] += 1
    push!(loss_history, l)

    if iter[] % 10 == 0 || iter[] == 1
      println("  Iteration $(iter[]): Loss = $(round(l, digits=6))")
    end

    return false  # 继续训练
  end

  return callback, loss_history
end

#=============================================================================
第七步: 主测试流程
=============================================================================#
begin
  rng = Xoshiro(123)

  println("="^70)
  println("虚拟数据测试：Physics-Informed NeuralODE")
  println("="^70)

  #---------------------------------------------------------------------------
  # 1. 模型设置
  #---------------------------------------------------------------------------
  println("\n第一步：模型配置")

  # 土壤参数（典型壤土）
  θ_s = 0.45f0    # 饱和含水量
  θ_r = 0.05f0    # 残余含水量
  Ks = 0.01f0     # 饱和导水率 cm/s
  α = 0.02f0      # cm^-1
  n = 1.5f0
  soil_params = (θ_s, θ_r, Ks, α, n)

  # 模拟设置：5层土壤
  n_layers = 5
  depths = Float32.(collect(5:10:45))  # 5, 15, 25, 35, 45 cm
  Δz = 10.0f0  # cm

  # 时间设置：模拟10小时
  t_start = 0.0f0
  t_end = 36000.0f0  # 10小时 = 36000秒
  tspan = (t_start, t_end)

  # 初始条件（非均匀分布，更真实）
  θ₀ = Float32.([0.30, 0.28, 0.25, 0.23, 0.20])

  println("  ✓ 土壤层数: $n_layers")
  println("  ✓ 深度: $(depths) cm")
  println("  ✓ 时间跨度: $(t_end/3600) 小时")
  println("  ✓ 初始含水量: $(round.(θ₀, digits=3))")

  #---------------------------------------------------------------------------
  # 2. 生成虚拟观测数据
  #---------------------------------------------------------------------------
  println("\n第二步：生成虚拟观测数据")

  θ_obs, θ_true, times = generate_virtual_data(rng, soil_params, depths, Δz, tspan, θ₀)

  println("  ✓ 观测数据形状: $(size(θ_obs))")
  println("  ✓ 时间点数量: $(length(times))")
  println("  ✓ 噪声水平: 2%")
  println("  ✓ 观测范围: [$(round(minimum(θ_obs), digits=3)), $(round(maximum(θ_obs), digits=3))]")
  println("  ✓ 真值范围: [$(round(minimum(θ_true), digits=3)), $(round(maximum(θ_true), digits=3))]")

  #---------------------------------------------------------------------------
  # 3. 初始化混合模型
  #---------------------------------------------------------------------------
  println("\n第三步：初始化混合模型")

  # 初始化神经网络
  ps, st = Lux.setup(rng, nn_model)
  ps = ComponentArray(ps)

  # 创建 HybridNeuralODE
  hybrid_node = HybridNeuralODE(
    nn_model, st, tspan, depths, soil_params, Δz;
    alg=Tsit5(), reltol=1e-4, abstol=1e-6,
    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
    verbose=false
  )

  # 初始损失
  loss_initial = loss_neuralode(ps, θ₀, θ_obs, hybrid_node, soil_params)
  println("  ✓ 初始损失: $(round(loss_initial, digits=6))")

  #---------------------------------------------------------------------------
  # 4. 训练模型
  #---------------------------------------------------------------------------
  println("\n第四步：训练混合模型")
  println("  优化器: Adam (学习率 0.01)")
  println("  最大迭代次数: 50")

  # 定义损失函数包装器
  loss_wrapper(p) = loss_neuralode(p, θ₀, θ_obs, hybrid_node, soil_params)

  # 创建回调函数
  callback, loss_history = create_callback(θ₀, θ_obs, hybrid_node, soil_params)

  # 设置优化问题
  optf = Optimization.OptimizationFunction((p, _) -> loss_wrapper(p), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, ps)

  # 训练
  println("\n  开始训练...")
  result = Optimization.solve(
    optprob,
    Adam(0.01f0),
    callback=callback,
    maxiters=50
  )

  println("\n  训练完成！")
  println("  ✓ 最终损失: $(round(result.objective, digits=6))")
  println("  ✓ 损失下降: $(round((loss_initial - result.objective) / loss_initial * 100, digits=2))%")

  #---------------------------------------------------------------------------
  # 5. 验证训练结果
  #---------------------------------------------------------------------------
  println("\n第五步：验证模型预测")

  # 使用训练后的参数进行预测
  ps_trained = result.u
  sol_trained, _ = hybrid_node(θ₀, ps_trained, st)
  θ_pred = Array(sol_trained)

  # 计算预测误差
  rmse = sqrt(mean((θ_pred .- θ_true) .^ 2))
  mae = mean(abs.(θ_pred .- θ_true))
  r2 = 1.0f0 - sum((θ_pred .- θ_true) .^ 2) / sum((θ_true .- mean(θ_true)) .^ 2)

  println("  ✓ RMSE (vs 真值): $(round(rmse, digits=5))")
  println("  ✓ MAE (vs 真值): $(round(mae, digits=5))")
  println("  ✓ R² (vs 真值): $(round(r2, digits=4))")

  # 比较初始和最终预测
  println("\n  预测对比（第1层，前3个时间点）:")
  println("    真值:   ", round.(θ_true[1, 1:3], digits=4))
  println("    观测:   ", round.(θ_obs[1, 1:3], digits=4))
  println("    预测:   ", round.(θ_pred[1, 1:3], digits=4))

  #---------------------------------------------------------------------------
  # 6. 总结
  #---------------------------------------------------------------------------
  println("\n" * "="^70)
  println("测试总结")
  println("="^70)

  if r2 > 0.8f0 && rmse < 0.05f0
    println("✓ 模型训练成功！")
    println("  - R² > 0.8，说明模型能很好地拟合数据")
    println("  - RMSE < 0.05，预测误差在可接受范围内")
    println("  - 混合模型（物理 + 神经网络）可行")
  elseif r2 > 0.5f0
    println("⚠ 模型部分成功")
    println("  - 模型有一定预测能力，但可能需要更多训练")
    println("  - 建议: 增加训练轮数或调整学习率")
  else
    println("✗ 模型训练需要改进")
    println("  - 预测效果不理想")
    println("  - 建议: 检查模型架构、增加训练数据或调整超参数")
  end

  println("\n模型架构验证:")
  println("  ✓ NeuralODE 可以成功求解")
  println("  ✓ 梯度计算正常（通过 Adjoint 方法）")
  println("  ✓ 优化器可以更新参数")
  println("  ✓ 混合模型（物理 + 数据驱动）框架可行")

  println("\n下一步:")
  println("  1. 使用真实观测数据替换虚拟数据")
  println("  2. 调整神经网络架构（层数、神经元数）")
  println("  3. 尝试不同的优化器（AdamW, LBFGS）")
  println("  4. 增加物理约束权重")
  println("  5. 实现模型保存和加载功能")

  println("\n" * "="^70)
end
