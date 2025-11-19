import RTableTools: fread, cbind
# using RCall

f_SM_Batesville = joinpath(@__DIR__, "../data", "SM_AR_Batesville_8_WNW_2024.csv") |> abspath

begin
  # 5层：5, 10, 20, 50, 100 cm
  d = fread(f_SM_Batesville)
  Prcp_cm = d.P_CALC ./ 10 # [mm] to [cm]
  yobs_full = d[:, 3:end] |> Matrix #|> drop_missing

  θ_t0 = yobs_full[1, 1:end] # t0
  θ_surf = yobs_full[:, 1] # 5cm
end
