using OffsetArrays

"""
    soil_depth_init(Δz::AbstractVector)
    
Soil depth initialization

```julia
z, z₋ₕ, z₊ₕ, Δz₊ₕ = soil_depth_init(Δz)
```
"""
function soil_depth_init(Δz::AbstractVector)
  # Soil depth (m) at i+1/2 interface between layers i and i+1 (negative distance from surface)
  # z_{i+1/2}
  N = length(Δz)
  z = OffsetArray(zeros(N + 2), 0:N+1)    # 虚拟层N+1
  # Δz₊ₕ = OffsetArray(zeros(N + 1), 0:N)
  Δz₊ₕ = zeros(N + 1)
  z₊ₕ = zeros(N + 1)
  z₋ₕ = zeros(N + 1)

  z₊ₕ[1] = -Δz[1]
  for i = 2:N
    z₊ₕ[i] = z₊ₕ[i-1] - Δz[i] # on the edge
  end

  # Soil depth (m) at center of layer i (negative distance from surface)
  z[1] = 0.5 * z₊ₕ[1]
  for i = 2:N
    z[i] = 0.5 * (z₊ₕ[i-1] + z₊ₕ[i]) # on the center
  end

  # Thickness between between z(i) and z(i+1)
  # Δz₊ₕ[0] = 0.5 * Δz[1]
  for i = 1:N-1
    Δz₊ₕ[i] = z[i] - z[i+1]
  end
  Δz₊ₕ[N] = 0.5 * Δz[N]

  ## z₋ₕ
  z₋ₕ[1] = 0
  z₋ₕ[2:end] = z₊ₕ[1:end-1]
  (; z, z₋ₕ, z₊ₕ, Δz₊ₕ)
end

function cal_Δz(z)
  N = length(z)
  z₊ₕ = zeros(N)
  Δz = zeros(N)
  Δz[1] = 0 - z[1] * 2
  z₊ₕ[1] = -Δz[1]

  for i in 2:N
    Δz[i] = (z₊ₕ[i-1] - z[i]) * 2
    z₊ₕ[i] = z₊ₕ[i-1] - Δz[i]
  end
  Δz
end

export soil_depth_init, cal_Δz
