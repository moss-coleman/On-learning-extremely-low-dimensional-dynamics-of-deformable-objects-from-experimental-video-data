using Flux
using Flux: throttle, params, Chain, Dense, pullback, MaxPool
using Flux: @epochs, mse, logitbinarycrossentropy, binarycrossentropy, crossentropy, logitcrossentropy
using Parameters
using Parameters: @with_kw
using VideoIO
using AbbreviatedStackTraces
using Images
using Random
using CUDA
using Flux.Data: DataLoader
using BSON: @save, @load



@with_kw mutable struct Args
    η = 1e-4                # learning rate
    λ = 0.01f0              # regularization paramater
    epochs = 5          # number of epochs
    seed = 42               # random seed
    cuda = true             # use GPU
    input_dim = [36, 64, 3] # image size
    latent_dim = 1          # latent dimension
    hidden_dim = 128        # hidden dimension
    tblogger = true         # log training with tensorboard
    beta = 0.5              # β value in loss function
    variation = "shape"    # what to vary
    data_set = "concave"      # what set of states to train the VAE to represent 
    filter_width = 4        # CNN kernal dimension

end

struct Reshape
    shape
end

Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()



function load_data(; kws...)
    args = Args(; kws...)
    println("../data/video/states/$(args.variation)_variation/states_$(args.data_set)_36x64.mp4")
    video = VideoIO.load("../data/video/states/$(args.variation)_variation/states_$(args.data_set)_36x64.mp4")
    frames = size(video, 1)
    x_data = zeros(Float32, frames, args.input_dim[3], args.input_dim[1], args.input_dim[2])
    for i in 1:frames
        x_data[i, 1, :, :] .= Float32.(channelview(Gray.(video[i])))
    end

    X_train = permutedims(x_data, [3, 4, 2, 1])

    return X_train
end

function vae_train!(encoder_μ, encoder_logvar, decoder, model_loss, x_train, ps; kws...)

    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.cuda && CUDA.functional()
    #println("checking device")
    if use_cuda
        device = gpu
        println("Using GPU")
    else
        device = cpu
        println("Using CPU")
    end


    encoder_μ |> device
    encoder_logvar |> device
    decoder |> device

    len_data_loader = size(x_train, 4)
    data_loader = DataLoader(x_train, batchsize=len_data_loader, shuffle=true)

    opt = ADAM(args.η)
    β = args.beta
    λ = args.λ

    train_steps = 0
    train_save = zeros(args.epochs)

    for epoch = 1:args.epochs

        n = 0
        l = 0
        for x in data_loader
            x |> device
            loss, back = pullback(ps) do
                model_loss(encoder_μ, encoder_logvar, decoder, x, β, λ)
            end
            grad = back(1.0f0)
            Flux.update!(opt, ps, grad)
            l += loss
        end

        avg_loss = l / len_data_loader

        train_save[epoch] = avg_loss
        if epoch % 10 == 0
            println("Epoch $epoch : Train loss = $(train_save[epoch]) ")
        end
    end



    encoder_μ |> cpu
    encoder_logvar |> cpu
    decoder |> cpu

    @save "../models/$(args.variation)_variation/encoder_mu_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10).bson" encoder_μ
    @save "../models/$(args.variation)_variation/encoder_logvar_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10).bson" encoder_logvar
    @save "../models/$(args.variation)_variation/decoder_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10).bson" decoder
end




function create_VAE(; kws...)
    args = Args(; kws...)

    pix_h = args.input_dim[1]
    pix_w = args.input_dim[2]
    channel = args.input_dim[3]
    Dz = 1
    cnn_output_size = Int.(floor.([pix_w / 8, pix_h / 8, 32]))
    # cnn_output_size = Int.(floor.([5, 8, 32])) # for (3, 3) filter
    # cnn_output_size = Int.(floor.([64, 36, 32])) # for (3, 3) filter
    # cnn_flat_size = Int.(floor((pix_w / 8) * (pix_h / 8) * 32))
    cnn_flat_size = 1024

    D_flat = 256
    # struct Reshape
    #     shape
    # end

    # Reshape(args...) = Reshape(args)
    # (r::Reshape)(x) = reshape(x, r.shape)
    # Flux.@functor Reshape ()

    filter_width = 4

    conv_1 = Conv((args.filter_width, args.filter_width), channel => 32, tanh; stride=2, pad=1) #in [480, 640, 3, 71] out -> [240, 320, 32, 71] = [h/stride, w/stride, 32, b]
    conv_2 = Conv((args.filter_width, args.filter_width), 32 => 32, tanh; stride=2, pad=1) # in [240, 320, 32, 71], out -> [120, 160, 32, 71] = [(h/stride)/stride, (w/stride)/stride, 32, b]
    conv_3 = Conv((args.filter_width, args.filter_width), 32 => 32, tanh; stride=2, pad=1) # in [120, 240, 32, 71], out -> [60, 80, 32, 71] = [((h/stride)/stride)/stride, ((w/stride)/stride)/stride, 32, b]

    dense_in_1 = Dense(cnn_flat_size, D_flat, tanh)

    encoder_features = Chain(Flux.flatten, Dense(pix_h * pix_w * channel, cnn_flat_size, tanh), dense_in_1)

    encoder_μ = Chain(encoder_features, Dense(D_flat, Dz))
    encoder_logvar = Chain(encoder_features, Dense(D_flat, Dz))


    dense_out_1 = Dense(Dz, D_flat, tanh)
    dense_out_2 = Dense(D_flat, cnn_flat_size, tanh)


    # dense decoder instead of conv decoder

    # dense_out_3 = Dense(cnn_flat_size, pix_h * pix_w * channel)

    # decoder = Chain(dense_out_1, dense_out_2, dense_out_3, Reshape(pix_h, pix_w, channel, :))
    #emphasise decoder = Chain(dense_out_1, dense_out_2, dense_out_3, x -> reshape(x, (pix_h, pix_w, channel, :)))

    #x -> reshape(x,(4,4,32,:))
    #reshape_out = Reshape(4, 4, 32, :)
    #deconv_1 = ConvTranspose((3, 3), 7*7*32=>64, re    # So we minimise the sum of the negative ELBO and a weight penalty

    deconv_1 = ConvTranspose((filter_width, filter_width), 32 => 32, tanh; stride=2, pad=(1, 1, 1, 1))
    deconv_2 = ConvTranspose((filter_width, filter_width), 32 => 32, tanh; stride=2, pad=(0, 0, 1, 1))
    deconv_out = ConvTranspose((filter_width, filter_width), 32 => channel; stride=2, pad=(1, 1, 1, 1))

    decoder = Chain(dense_out_1, dense_out_2, Reshape(cnn_output_size[2], cnn_output_size[1], 32, :), deconv_1, deconv_2, deconv_out)
    return encoder_μ, encoder_logvar, decoder
end



function vae_loss(encoder_μ, encoder_logvar, decoder, x, β, λ; kws...)
    # args = Args(; kws...)
    batch_size = size(x)[end]
    @assert batch_size != 0
    μ = encoder_μ(x)
    logvar = encoder_logvar(x)

    # reparameterisation 
    z = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)

    x̂ = decoder(z)

    logp_x_z = -(logitbinarycrossentropy(x̂, x; agg=sum)) / batch_size

    kl_q_p = 0.5f0 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0f0)) / batch_size

    reg = λ * sum(x -> sum(x .^ 2), Flux.params(encoder_μ, encoder_logvar, decoder))

    elbo = logp_x_z - β .* kl_q_p

    return -elbo + reg
end


state_data = load_data()
encoder_μ, encoder_logvar, decoder = create_VAE()

vae_ps = Flux.params(encoder_μ, encoder_logvar, decoder)
# function vae_train!(encoder_μ, encoder_logvar, model_loss, x_train, ps; kws...)
vae_train!(encoder_μ, encoder_logvar, decoder, vae_loss, state_data, vae_ps)



