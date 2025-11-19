export richards_dθdt, HybridNeuralODE
export loss_mse, train, predict, evaluate

using Optimization, OptimizationOptimisers


# Richards方程：计算达西通量和含水量变化
# 单位：P_flux [cm/h], K [cm/h], depths [cm], dθdt [1/h]
function richards_dθdt(θ, depths, soil_params_profile, P_flux=0.0f0)
  n = length(θ)
  dθdt = zeros(eltype(θ), n)

  # 计算每层的水力特性 [cm/h]
  K = zeros(eltype(θ), n)
  ψ = zeros(eltype(θ), n)

  for i in 1:n
    θ_s, θ_r, Ks, α, n_vg = soil_params_profile[i]
    θ_safe = clamp(θ[i], θ_r + 0.001f0, θ_s - 0.001f0)

    K[i] = hydraulic_conductivity(θ_safe, θ_s, θ_r, Ks, n_vg)  # [cm/h]
    ψ[i] = soil_water_potential(θ_safe, θ_s, θ_r, α, n_vg)     # [cm]
  end

  # 计算界面水力传导度和达西通量 [cm/h]
  Q = zeros(eltype(θ), n + 1)

  # 上边界：降水入渗 [cm/h]
  Q[1] = P_flux

  # 内部节点：达西定律 Q = -K·(∂ψ/∂z + 1) [cm/h]
  for i in 2:n
    Δz = depths[i] - depths[i-1]  # [cm]
    K_interface = 2.0f0 * K[i-1] * K[i] / (K[i-1] + K[i] + Float32(1e-10))  # 调和平均 [cm/h]
    ψ_gradient = (ψ[i] - ψ[i-1]) / Δz  # [cm/cm] = [-]
    Q[i] = -K_interface * (ψ_gradient + 1.0f0)  # [cm/h]
  end

  # 下边界：自由排水 [cm/h]
  Q[n+1] = -K[n]

  # 计算含水量变化：∂θ/∂t = -∂Q/∂z [1/h]
  for i in 1:n
    Δz_cell = if i == 1
      depths[2] - depths[1]
    elseif i == n
      depths[n] - depths[n-1]
    else
      (depths[i+1] - depths[i-1]) / 2.0f0
    end  # [cm]
    dθdt[i] = -(Q[i+1] - Q[i]) / Δz_cell  # [cm/h] / [cm] = [1/h]
  end

  return dθdt
end


# 创建混合ODE函数（Richards + Neural correction）
function create_hybrid_ode(nn_model, nn_state, depths, soil_params_profile, P_interp)
  function ode!(dθ, θ, p, t)
    # 获取当前时刻降水通量（需要支持Dual numbers for autodiff）
    t_value = float(t)  # Convert Dual to Float if needed
    P_flux = P_interp(t_value)

    # 安全限制θ范围
    θ_s_mean = mean([sp[1] for sp in soil_params_profile])
    θ_r_mean = mean([sp[2] for sp in soil_params_profile])
    θ_safe = clamp.(θ, θ_r_mean + 0.001f0, θ_s_mean - 0.001f0)

    # Richards方程物理项
    dθ_physics = richards_dθdt(θ_safe, depths, soil_params_profile, P_flux)

    # 神经网络残差修正
    nn_input = vcat(reshape(θ_safe, :, 1), fill(P_flux, 1, 1))  # [θ; P]
    corrections, _ = nn_model(nn_input, p, nn_state)
    dθ_nn = vec(corrections)

    # 混合动力学（物理主导，NN小幅修正）
    dθ_hybrid = dθ_physics .+ 0.001f0 .* dθ_nn

    # 限制变化率防止数值爆炸
    dθ .= clamp.(dθ_hybrid, -0.001f0, 0.001f0)
    return nothing
  end
  return ode!
end


# NeuralODE 包装器
struct HybridNeuralODE{M,T,P,S,F}
  nn_model::M
  nn_state::S
  tspan::T
  depths::Vector{Float32}
  soil_params_profile::P  # Vector of tuples: [(θ_s, θ_r, Ks, α, n) for each layer]
  P_interp::F  # 降水插值函数
  alg
  kwargs
end


function HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params_profile, P_interp;
                         alg=Tsit5(), kwargs...)
  return HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params_profile,
                        P_interp, alg, kwargs)
end


function (node::HybridNeuralODE)(θ₀, p, st)
  ode_func! = create_hybrid_ode(node.nn_model, node.nn_state, node.depths,
                                node.soil_params_profile, node.P_interp)
  prob = ODEProblem(ode_func!, θ₀, node.tspan, p)

  # tspan现在是小时，每小时保存一次
  n_save = Int(node.tspan[2] - node.tspan[1])
  saveat = range(node.tspan[1], node.tspan[2], length=n_save+1)[1:end-1]
  sol = solve(prob, node.alg; node.kwargs..., saveat=saveat)
  return sol, st
end
