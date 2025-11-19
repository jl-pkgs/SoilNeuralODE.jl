export diffusion_dθdt, HybridNeuralODE
export loss_mse, train, predict, evaluate

using Optimization, OptimizationOptimisers

# 简化扩散模型
function diffusion_dθdt(θ, Δz; D=0.0001f0)
  n = length(θ)
  dθdt = zeros(eltype(θ), n)
  for i in 2:n-1
    dθdt[i] = D * (θ[i+1] - 2θ[i] + θ[i-1]) / (Δz * Δz)
  end
  return dθdt
end

# 创建混合ODE函数
function create_hybrid_ode(nn_model, nn_state, soil_params, Δz)
  function ode!(dθ, θ, p, t)
    θ_s, θ_r = soil_params[1:2]
    θ_safe = clamp.(θ, θ_r + 0.001f0, θ_s - 0.001f0)

    dθ_physics = diffusion_dθdt(θ_safe, Δz)
    corrections, _ = nn_model(reshape(θ_safe, :, 1), p, nn_state)
    dθ_hybrid = dθ_physics .+ 0.00001f0 .* vec(corrections)

    dθ .= clamp.(dθ_hybrid, -0.0001f0, 0.0001f0)
    return nothing
  end
  return ode!
end

# NeuralODE 包装器
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

function HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params, Δz; alg=Tsit5(), kwargs...)
  return HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params, Δz, alg, kwargs)
end

function (node::HybridNeuralODE)(θ₀, p, st)
  ode_func! = create_hybrid_ode(node.nn_model, node.nn_state, node.soil_params, node.Δz)
  prob = ODEProblem(ode_func!, θ₀, node.tspan, p)

  n_save = Int((node.tspan[2] - node.tspan[1]) / 3600.0f0)
  saveat = range(node.tspan[1], node.tspan[2], length=n_save+1)[1:end-1]
  sol = solve(prob, node.alg; node.kwargs..., saveat=saveat)
  return sol, st
end

# MSE损失函数
function loss_mse(p, θ₀, θ_obs, node)
  sol, _ = node(θ₀, p, NamedTuple())
  θ_pred = Array(sol)
  return mean((θ_pred .- θ_obs) .^ 2)
end

# 预测函数
function predict(node, θ₀, p)
  sol, _ = node(θ₀, p, NamedTuple())
  return Array(sol)
end

# 评估函数：对每层计算GOF
function evaluate(node, θ₀, p, θ_obs; to_df=true)
  θ_pred = predict(node, θ₀, p)
  n_layers = size(θ_obs, 1)

  gofs = []
  for i in 1:n_layers
    gof = GOF(θ_obs[i, :], θ_pred[i, :])
    push!(gofs, gof)
  end

  !to_df && return gofs

  depths = node.depths
  DataFrame([
    (; layer=i, depth=depths[i], gofs[i]...) for i in 1:n_layers
  ])
end

# 训练函数
function train(node, ps, θ₀, θ_obs;
  nepoch=50, step=10, lr=0.001f0)

  loss_wrapper(p) = loss_mse(p, θ₀, θ_obs, node)

  iter = Ref(0)
  callback(state, _) = (
    iter[] += 1;
    if iter[] % step == 0
      gof = evaluate(node, θ₀, state.u, θ_obs; to_df=false)
      nse_mean = mean([g.NSE for g in gof])
      println("Epoch $(lpad(iter[], 3)) | NSE = $(round(nse_mean, digits=6))")
    end;
    false
  )

  optf = Optimization.OptimizationFunction((p, _) -> loss_wrapper(p), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, ps)

  @info "Start training:"
  result = Optimization.solve(optprob, Adam(lr), callback=callback, maxiters=nepoch)

  gof = evaluate(node, θ₀, result.u, θ_obs)
  node, result.u, gof
end
