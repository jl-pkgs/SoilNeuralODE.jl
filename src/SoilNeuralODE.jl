module SoilNeuralODE

export Soil, create_soil_profile
export hydraulic_conductivity, soil_water_potential

# Physics-Informed NeuralODE for soil moisture simulation
using Lux, DifferentialEquations, ComponentArrays
using SciMLSensitivity, Zygote
using Statistics
using ModelParams: GOF
using DataFrames
using Parameters: @with_kw


include("Soil.jl")
include("NeuralODE.jl")
include("Framework.jl")
include("soil_depth_init.jl")


end # module SoilNeuralODE
