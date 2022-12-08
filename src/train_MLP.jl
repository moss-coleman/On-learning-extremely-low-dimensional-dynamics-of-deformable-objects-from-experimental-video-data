using Revise
using Flux: throttle, params, Chain, Dense
using Flux, Statistics
using Flux.Data: DataLoader
using Random
using CUDA
using Parameters: @with_kw
using BSON: @save, @load




@with_kw mutable struct Args_mlp
    η = 1e-4                # learning rate
    λ = 0.01f0              # regularization paramater
    epochs = 5              # number of epochs
    seed = 42               # random seed
    cuda = true             # use GPU
    input_dim = 20          # image size
    latent_dim = 1          # latent dimension
    hidden_layers = 20      # latent dimension
    beta = 1.0              # β value in loss function
    variation = "shape"     # what to vary, "shape" or "length"
    data_set = "concave"    # what set of states to train the VAE to represent
end


function load_encoded_data(; kws...)

    args = Args_mlp(; kws...)
    @load "../data/encoded_time_series/$(args.variation)_variation/x_train_$(args.data_set).bson" x_train
    @load "../data/encoded_time_series/$(args.variation)_variation/x_test_$(args.data_set).bson" x_test
    @load "../data/encoded_time_series/$(args.variation)_variation/y_train_$(args.data_set).bson" y_train
    @load "../data/encoded_time_series/$(args.variation)_variation/y_test_$(args.data_set).bson" y_test
    return x_train, x_test, y_train, y_test
end


x_train, x_test, y_train, y_test = load_encoded_data()


function mlp_train!(model, model_loss, x_data, y_data, ps; kws...)
    args = Args_mlp(; kws...)

    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    opt = Flux.Adam(args.η)

    @info "MLP: $(num_params(model)) trainable params"

    train_save = zeros(args.epochs)

    data_loader = DataLoader((x_data, y_data), batchsize=128, shuffle=true)

    for epoch = 1:args.epochs
        n = 0
        l = 0

        for (x, y) in data_loader
            x |> device
            y |> device
            loss, back = Flux.pullback(ps) do
                model_loss(x, y)
            end
            grad = back(1.0f0)
            Flux.update!(opt, ps, grad)
            l += loss
            n += size(x)[end]
        end
        train_save[epoch] = l / n

        if epoch % 100 == 0
            @info "Epoch $epoch : Train loss = $(train_save[epoch]) "
        end

    end

    model |> cpu
    @save "../models/$(args.variation)_variation/mlp_$(args.data_set).bson" model

end

function create_MLP(; kws...)
    args = Args_mlp(; kws...)


    mlp = Chain(Dense(args.input_dim, args.hidden_layers, tanh),
        Dense(args.hidden_layers, args.hidden_layers, tanh),
        Dense(args.hidden_layers, args.hidden_layers, tanh),
        Dense(args.hidden_layers, args.hidden_layers, tanh),
        Dense(args.hidden_layers, args.latent_dim))

    return mlp
end

num_params(model) = sum(length, Flux.params(model))

mse_loss(x, y) = Flux.mse(mlp(x), y)
# λ = 0.01f0
# # function mse_loss(x, y)
# #     mse = Flux.mse(mlp(x), y)
# #     reg = λ * sum(x -> sum(x .^ 2), Flux.params(mlp))
# #     return mse + reg
# # end

mlp = create_MLP()

ps = Flux.params(mlp)

mlp_train!(mlp, mse_loss, x_train, y_test, ps)




