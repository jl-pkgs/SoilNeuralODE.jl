using Lux, Enzyme, ComponentArrays, Optimisers, Random

# 定义简单模型：输入1维，输出1维
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

# 生成数据（模拟土壤水数据，x 为深度，y 为含水量）
rng = Xoshiro(123)
X = rand(rng, Float32, 1, 100) .* 10.0f0  # 深度 0-10
y = sin.(X) .+ 0.1f0 * randn(rng, 1, 100)  # 模拟含水量
y = Float32.(y)


# 损失函数（MSE）
function loss_func(ps, X, y, model, st)
  # ps .= ps_flat
  # ps = ComponentArray(ps_flat, getaxes(ps))  # 从扁平向量恢复 ComponentArray
  y_pred, _ = model(X, ps, st)
  return mean((y_pred .- y) .^ 2)
end

begin
  ## 方案2: 更佳
  ps, st = Lux.setup(rng, model)
  ps = ComponentArray(ps)
  dps = make_zero(ps)
  @time Enzyme.autodiff(Reverse, loss_func,
    Duplicated(ps, dps), Const(X), Const(y), Const(model), Const(st))
  println(dps)  # 检查梯度值是否合理（非零、非 NaN）
end



begin
  ## 方案1
  ps, st = Lux.setup(rng, model)
  dps = make_zero(ps)

  @time Enzyme.autodiff(Reverse, loss_func,
    Duplicated(ps, dps), Const(X), Const(y), Const(model), Const(st))
  println(dps)  # 检查梯度值是否合理（非零、非 NaN）
end

