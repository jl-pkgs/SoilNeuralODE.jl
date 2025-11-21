export AbstractSoilParam, VanGenuchten, Campbell
export AbstractHydraulics
export VanGenuchtenLayers, CampbellLayers
export SoilHydra, get_params
# export SoilParam, get_soilpar

using StructArrays

# 2.5x faster power method
"Faster method for exponentiation"
pow(x, y) = x^y
# @fastmath pow(x::Real, y::Real) = exp(y * log(x))


include("Retention.jl")
include("Retention_van_Genuchten.jl")
include("macro.jl")
# include("Retention_Campbell.jl")
# include("ParamTable.jl")
# include("Ponding.jl")

## Deprecated methods
abstract type AbstractHydraulics{FT,N} end

@make_layers_struct VanGenuchten VanGenuchtenLayers AbstractHydraulics
@make_layers_struct Campbell CampbellLayers AbstractHydraulics
