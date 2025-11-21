using SciMLSensitivity
import Base.zero

struct Dense{T,F<:Function}
  n_inp::Int
  n_nodes::Int
  W::Matrix{T}
  b::Vector{T}
  activation::F
end

function Dense(n_inp, n_nodes, f::Function, T, randfn::Function=rand)
  Dense(n_inp, n_nodes, convert.(T, randfn(n_nodes, n_inp)),
    convert.(T, randfn(n_nodes)), f)
end

struct NN{T}
  n_inp::Int
  layers::Vector{Dense{T}}
  intermediates::Vector{Vector{T}}
end

function NN(n_inp, layers, ::Type{T}) where {T}
  @assert length(layers) >= 1
  @assert n_inp == layers[1].n_inp
  for i in eachindex(layers)[1:(end-1)]
    @assert layers[i].n_nodes == layers[i+1].n_inp
  end
  NN(n_inp, layers, [zeros(T, layer.n_nodes) for layer in layers])
end

function paramlength(nn::NN)
  r = 0
  for l in nn.layers
    r = r + length(l.W)
    r = r + length(l.b)
  end
  return r
end

function get_params(nn::NN)
  ret = eltype(nn.layers[1].W)[]
  for l in nn.layers
    append!(ret, l.W)
    append!(ret, l.b)
  end
  return ret
end

function set_params(nn, params)
  i = 1
  for l in nn.layers
    l.W .= reshape(params[i:(i+length(l.W)-1)], size(l.W))
    i = i + length(l.W)
    l.b .= params[i:(i+length(l.b)-1)]
    i = i + length(l.b)
  end
end
