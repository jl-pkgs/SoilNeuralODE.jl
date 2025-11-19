# 训练函数
function train(node, ps, θ₀, θ_obs;
  nepoch=50, step=10, lr=0.001f0)

  loss_wrapper(p) = loss_mse(p, θ₀, θ_obs, node)

  iter = Ref(0)
  callback(state, _) = (
    iter[] += 1;
    if iter[] % step == 0
      gof = evaluate(node, θ₀, state.u, θ_obs; to_df=false)
      value = mean([g.MAE for g in gof])
      println("Epoch $(lpad(iter[], 3)) | MAE = $(round(value, digits=6))")
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
  θ_pred = predict(node, θ₀, p) # 多层的结果
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
