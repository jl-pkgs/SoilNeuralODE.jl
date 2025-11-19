## 添加一个简单的绘图函数
begin
  using Plots
  gr(framestyle=:box)

  ysim = predict(node, θ₀, ps_trained)
  yobs = θ_obs


  function plot_layer(i)
    plot(ysim[i, :], label="Simulated", lw=2)
    scatter!(yobs[i, :], label="Observed", ms=3)
    xlabel!("Time (hours)")
    ylabel!("Soil Moisture Layer $(depths[i]) cm")
  end

  ps = map(plot_layer, 1:size(θ_obs, 1))
  plot(ps..., layout=(2, 3), size=(600, 800))
  savefig("soil_moisture_comparison.png")
end
