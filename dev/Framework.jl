using Random, Statistics
using Flux
using MLUtils  # DataLoader
using Functors
using UnPack
using ModelParams
using Printf


function loss_mse(m, x, y)
  # 每个 batch 重置状态，避免跨 batch 泄漏
  Flux.reset!(m)                    # ! no need any more
  ŷ = m(x) |> vec                   # (1, B) —— LSTM 最后时刻输出 → Dense → 1×B
  return Flux.Losses.mse(ŷ, y)
end


"""
# Arguments
- `X`: time in the last dimension by default
- `fmt`: format string for printing NSE values, default is "%.6f"
"""
function train(model, X, Y;
  nepoch=400, step=10, device=cpu,
  loss::Function=loss_mse,
  perc_train=0.7, perc_valid=0.15, batchsize=128, DimTime_X=0, fmt="%.6f")

  (DimTime_X == 0) && (DimTime_X = ndims(X))

  ntime = length(Y)
  index = split_index(ntime; perc_train, perc_valid)
  data = split_forcing(X, Y, index; device, DimTime_X)

  loader_train = DataLoader(data.train; batchsize, shuffle=false)
  
  opt = Adam(1e-3)
  opt_state = Flux.setup(opt, model)

  @info "Start training:"
  for epoch in 1:nepoch
    lsum = 0.0
    nbt = 0
    # epoch_start = time()
    for (xb, yb) in loader_train
      gs = gradient(model) do m
        loss(m, xb, yb)
      end
      Flux.update!(opt_state, model, gs[1]) # Use gs[1] as per the warning
      lsum += loss(model, xb, yb) |> float
      nbt += 1
    end
    # epoch_time = time() - epoch_start
    # avg_loss = lsum / nbt
    # 每个epoch都显示简单进度
    # print("\rEpoch $epoch/$nepoch | Loss: $(round(avg_loss, digits=4))")

    if mod(epoch, step) == 0
      Flux.testmode!(model)
      gof = evaluate(model, data; to_df=false)

      format_str = "Epoch %03d, NSE | train = $fmt, valid = $fmt, test = $fmt \n"
      @eval @printf($format_str, $epoch, $(gof.train.NSE), $(gof.valid.NSE), $(gof.test.NSE))
      Flux.trainmode!(model)
    end
  end
  gof = evaluate(model, data)
  data, model, gof
end


function predict(model, X)
  # Flux.testmode!(model)
  Flux.reset!(model)
  model(X) |> vec  # Y_sim
end

function evaluate(model, data; to_df=true)
  ŷ_train = predict(model, data.train.X)
  ŷ_valid = predict(model, data.valid.X)
  ŷ_test = predict(model, data.test.X)

  gof_train = GOF(Array(data.train.Y[:]), Array(ŷ_train))
  gof_valid = GOF(Array(data.valid.Y[:]), Array(ŷ_valid))
  gof_test = GOF(Array(data.test.Y[:]), Array(ŷ_test))
  !to_df && return (; train=gof_train, valid=gof_valid, test=gof_test)

  DataFrame([
    (; type="train", gof_train...),
    (; type="valid", gof_valid...),
    (; type="test", gof_test...)
  ])
end

export loss_mse
export train, predict, evaluate
# export build_LSTM, train_LSTM, eval_LSTM
# export build_LSTM_embedding
