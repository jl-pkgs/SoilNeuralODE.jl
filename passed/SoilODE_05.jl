include("main_pkgs.jl")
include("main_forcing.jl")
using ADTypes


data_file = "data/SM_AR_Batesville_8_WNW_2024.csv"
θ_obs, P = load_soil_data(data_file)
inds = [2, 3, 4, 5, 6]

Q_t0 = θ_obs[:, 1]
yobs = θ_obs[:, 2:6*1+1]  # 去掉第一个时刻的观测值
ntime = size(yobs, 2)


begin
  # 供7层土壤
  z = -[1.25, 5, 10, 20, 50, 100.0, 200] ./ 100 # 第一层是虚拟的
  Δz = cal_Δz(z)
  z, z₋ₕ, z₊ₕ, Δz₊ₕ = soil_depth_init(Δz)
  n_layers = length(Δz)
  soil = Soil{Float64}(; n_layers, z, Δz₊ₕ)

  # 3. Problem Setup
  # Parameters: θ_s, θ_r, Ks, α, n
  p_true = ComponentArray(θ_s=0.45, θ_r=0.05, Ks=10.0, α=0.02, n=3.0)
  p_flat = collect(p_true)
  u0 = fill(0.2, n_layers)
  u0[inds] .= Q_t0
  u0[inds[end]+1:end] .= Q_t0[end]

  # Initial dummy problem (will be remade)
  tspan = (1, ntime)
  prob = ODE.ODEProblem((dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil), u0, tspan, p_flat)
  # alg = ODE.Rodas5P(autodiff=AutoEnzyme())
  alg = ODE.Tsit5()
  @time ysim = predict(u0, p_flat, soil, prob, tspan; saveat=1, alg)[inds, :]
end


## 2. 测试梯度求解
begin
  p = copy(p_flat)
  dp = make_zero(p)
  du0 = make_zero(u0)
  dsoil = make_zero(soil)

  fill!(dp, 0.0)
  fill!(du0, 0.0)
  tspan = (0, 1)

  @time Enzyme.autodiff(
    Enzyme.Reverse,
    loss_adjoint_enzyme,
    Enzyme.Active,
    Enzyme.Duplicated(u0, du0),
    # Enzyme.Const(u0),
    Enzyme.Duplicated(p, dp),
    Enzyme.Duplicated(soil, dsoil),
    Enzyme.Const(yobs),
    Enzyme.Const(prob),
    Enzyme.Const(tspan),
    # Enzyme.Const(ODE.Rodas5P(autodiff=false))
  )
  
  dp
  lr = 0.05
  opt = Optimisers.Adam(lr)
  opt_state = Optimisers.setup(opt, p)
  opt_state, p = Optimisers.update(opt_state, p, dp)
  p
end


Random.seed!(42)

include("main_pkgs.jl")
@time p_opt = train(p_flat, u0, soil, yobs, prob, tspan;
  epochs=3, lr=0.1, alg=ODE.Tsit5())

p = [0.469999999999481, 0.0699999999997886, 9.980000000065722, 1.0e-5, 1.9800000000015392]
p = [0.5499999999971423, 0.14999999999955893, 9.900000000269053, 1.0e-5, 2.9000000000156763]
ysim_opt = predict(u0, p, soil, prob, tspan; saveat=1)[inds, :]


begin
  ps = map(plot_layer, 1:5)
  p = plot(ps..., layout=(3, 2), size=(800, 600))
  savefig(p, "soil_moisture_optimization.png")
end
