using SciMLSensitivity, OrdinaryDiffEq
using QuadGK, Enzyme, Zygote, ForwardDiff, LinearAlgebra
using Test

include("main_pkgs.jl")

# 定义无状态层：只存储元数据
struct Dense{F<:Function}
  n_inp::Int
  n_out::Int
  activation::F
end

# 定义无状态网络：只存储层列表
struct NN{L}
  layers::L
  n_params::Int
end

function NN(layers::Vector{<:Dense})
  # 预计算参数总长度，避免运行时计算
  n_params = sum(l -> l.n_inp * l.n_out + l.n_out, layers)
  NN(layers, n_params)
end

# 获取初始参数 (只在初始化时使用一次)
function init_params(nn::NN, T=Float64)
  p = Vector{T}(undef, nn.n_params)
  idx = 1
  for l in nn.layers
    # Xavier/Glorot initialization simplified
    limit = sqrt(6 / (l.n_inp + l.n_out))
    len_W = l.n_inp * l.n_out
    p[idx:idx+len_W-1] .= 2 * limit .* rand(T, len_W) .- limit
    idx += len_W
    p[idx:idx+l.n_out-1] .= 0.0 # bias init
    idx += l.n_out
  end
  return p
end

# 纯函数式前向传播：直接使用 p，无副作用
# idx 是当前参数指针位置
function apply_layer(l::Dense, x, p, idx)
  # W 的大小是 (n_out, n_inp)
  len_W = l.n_out * l.n_inp
  # 使用 reshape 重构视图，不进行复制
  W = reshape(view(p, idx:idx+len_W-1), l.n_out, l.n_inp)
  idx += len_W

  b = view(p, idx:idx+l.n_out-1)
  idx += l.n_out

  # 计算输出: W*x + b
  # 注意：为了极简和AD兼容，这里如果不涉及庞大矩阵，直接计算即可
  # 对于小维度，编译器会优化。
  out = l.activation.(W * x .+ b)
  return out, idx
end

function (nn::NN)(x, p)
  state = x
  idx = 1
  for l in nn.layers
    state, idx = apply_layer(l, state, p, idx)
  end
  return state
end

# --- 问题设置 ---

const step_val = 0.22
const H_0 = diagm(1:2)
const H_D = 0.01 * [0.0 1.0; 1.0 0.0]
const repart = 1:2
const impart = 3:4

# 构造网络结构
nn_struct = NN([
  Dense(4, 10, tanh),
  Dense(10, 4, sin)
])

# 构造 ODE 函数
# 注意：这里不需要闭包捕获可变的 nn 状态，nn_struct 是常量的
function dfunc(dstate, state, p, t)
  # 物理部分
  # 使用 @views 减少切片分配
  @views begin
    dstate[impart] .= -1.0 .* (H_0 * state[repart] .+ cos(2.0 * t) .* H_D * state[repart])
    dstate[repart] .= H_0 * state[impart] .+ cos(2.0 * t) .* H_D * state[impart]
  end

  # 神经网络部分：直接计算增量
  nn_out = nn_struct(state, p)

  dstate .+= nn_out
  nothing
end

# --- 求解与测试 ---

y0 = [1.0, 0.0, 0.0, 0.0]
p = init_params(nn_struct, Float64)
tspan = (0.0, 20.0)

# 正向测试
ds = zero(y0)
dfunc(ds, y0, p, 0.2)

# 求解 ODE
prob = ODEProblem{true}(dfunc, y0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-10)

# 目标函数
const target = copy(y0)
target[2] = 1.0

function g(u, p, t)
  diff = u .- target
  dot(diff, diff) # 简单的 L2 loss
end

# 1. ForwardDiff 基准 (Ground Truth)
function gintegrate(p)
  _prob = remake(prob, p=p)
  _sol = solve(_prob, Tsit5(), abstol=1e-12, reltol=1e-12)
  # 使用 QuadGK 积分损失
  res, err = quadgk(t -> g(_sol(t), p, t), tspan[1], tspan[2])
  return res
end

println("Calculating ForwardDiff Reference...")
refdp = ForwardDiff.gradient(gintegrate, p)

# 2. Adjoint Sensitivities 测试
println("Calculating Adjoint Sensitivities...")

# BacksolveAdjoint
du1, dp1 = adjoint_sensitivities(sol, Tsit5(), g=g,
  sensealg=BacksolveAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
  abstol=1e-12, reltol=1e-12)

println("BacksolveAdjoint Error: ", norm(dp1' - refdp))
@test isapprox(dp1', refdp, atol=1e-5)

# GaussAdjoint (通常更稳定)
du2, dp2 = adjoint_sensitivities(sol, Tsit5(), g=g,
  sensealg=GaussAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
  abstol=1e-12, reltol=1e-12)

println("GaussAdjoint Error: ", norm(dp2' - refdp))
@test isapprox(dp2', refdp, atol=1e-5)

# InterpolatingAdjoint
du4, dp4 = adjoint_sensitivities(sol, Tsit5(), g=g,
  sensealg=InterpolatingAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
  abstol=1e-12, reltol=1e-12)

println("InterpolatingAdjoint Error: ", norm(dp4' - refdp))
@test isapprox(dp4', refdp, atol=1e-5)

println("All tests passed cleanly.")
