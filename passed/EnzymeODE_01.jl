import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS
import Zygote
import Enzyme
import Enzyme: make_zero


function fiip(du, u, p, t)
  du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
  du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0; 1.0]
prob = ODE.ODEProblem(fiip, u0, (0.0, 10.0), p)

function loss(u0, p, prob; kw...)
  # sol = ODE.solve(prob, ODE.Tsit5(); u0, p, saveat=0.1, sensealg=sens_alg)
  new_prob = ODE.remake(prob, u0=u0, p=p)
  sol = ODE.solve(new_prob, ODE.Tsit5(); saveat=0.1, kw...)
  return sum(sol)
end


du0 = make_zero(u0)  # 用于存放 u0 的梯度
dp = make_zero(p)    # 用于存放 p 的梯度

# Active: 表示函数的返回值是“活性的”（即我们要对这个标量返回值求导，初始导数为1.0）
@time Enzyme.autodiff(
  Enzyme.Reverse,
  loss,
  Enzyme.Active,
  Enzyme.Duplicated(u0, du0),
  Enzyme.Duplicated(p, dp),
  Enzyme.Const(prob)
)

# 5. 查看结果
println("Gradient w.r.t u0: ", du0)
println("Gradient w.r.t p:  ", dp)

## SMS.InterpolatingAdjoint

sens_alg = SMS.InterpolatingAdjoint(autojacvec=SMS.ZygoteVJP())
# sol = ODE.solve(prob, ODE.Tsit5())
loss_adjoint_zygote(u0, p) = sum(ODE.solve(prob, ODE.Tsit5(); u0=u0, p=p, saveat=0.1, sensealg=sens_alg))
@time du0, dp = Zygote.gradient(loss_adjoint_zygote, u0, p)



function loss_adjoint_enzyme(u0, p, prob)
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())
  new_prob = ODE.remake(prob, u0=u0, p=p)
  sol = ODE.solve(new_prob, ODE.Tsit5(); saveat=0.1, sensealg)
  return sum(sol)
end

du0 = make_zero(u0)  # 用于存放 u0 的梯度
dp = make_zero(p)    # 用于存放 p 的梯度

# 4. 执行 Enzyme 自动微分
@time Enzyme.autodiff(
  Enzyme.Reverse,
  loss_adjoint_enzyme,
  Enzyme.Active,
  Enzyme.Duplicated(u0, du0),
  Enzyme.Duplicated(p, dp),
  Enzyme.Const(prob)
)

println("Gradient w.r.t u0: ", du0)
println("Gradient w.r.t p:  ", dp)
