# # # Plotting
# using Plots
# gr(framestyle=:box)
# function plot_layer(i)
#   plot(θ_obs[i, :], label="Observed", title="Soil Layer $(i)", xlabel="Time Step", ylabel="Soil Moisture θ")
#   plot!(ysim_opt[i, :], label="Optimized", lw=2)
#   # plot!(ysim[i, :], label="Initial", ls=:dash)
# end


## 其中观测数据在[5, 10, 20, 50, 100]，也即是: 
function load_soil_data(f)
  df = fread(f)
  # 提取土壤水分观测 (5层: 5, 10, 20, 50, 100 cm)
  θ_obs = Matrix(df[:, [:SOIL_MOISTURE_5, :SOIL_MOISTURE_10, :SOIL_MOISTURE_20,
    :SOIL_MOISTURE_50, :SOIL_MOISTURE_100]])
  # 提取降水 [mm] -> [cm]
  P = Float32.(df.P_CALC ./ 10.0)
  return θ_obs', P  # 转置为 (n_layers, n_times)
end
