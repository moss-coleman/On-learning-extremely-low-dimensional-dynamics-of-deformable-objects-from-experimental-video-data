using Revise
using VideoIO
using Flux
using Flux: throttle, params, Chain, Dense, unsqueeze
using Flux.Data: DataLoader
using BSON: @save, @load
using AbbreviatedStackTraces
using Parameters
using Parameters: @with_kw
using Images
using Statistics



# --- Load VAE Models --- # 
struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()



@with_kw mutable struct Args_data
    seed = 42               # random seed
    cuda = true             # use GPU
    input_dim = [36, 64, 1] # image size
    resize_ratio = 10
    latent_dim = 1          # latent dimension
    beta = 1.0              # β value in loss function
    variation = "shape"     # what to vary, "shape" or "length"
    data_set = "concave"    # what set of states to train the VAE to represent
end

function load_encoder(; kws...)
    args = Args_data(; kws...)
    @load "../models/$(args.variation)_variation/encoder_mu_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10).bson" encoder_μ
    return encoder_μ
end

function video2time_series(video, encoder; kws...)

    args = Args_data(; kws...)
    resize_ratio = args.resize_ratio
    frames = size(video, 1)
    pix_h = args.input_dim[1]
    pix_w = args.input_dim[2]
    channel = args.input_dim[3]
    video_resized = zeros(Float32, pix_h, pix_w, channel, frames)
    time_series_whole = zeros(Float32, frames)
    for i in 1:frames
        img = Float32.(channelview(Gray.(imresize(video[i], ratio=1 / resize_ratio))))
        video_resized[:, :, channel, i] .= img
    end

    time_series_whole[:] = encoder(video_resized)

    # create the training data 
    sample_interval = 1
    dt = (1.0 / 240.0) * sample_interval
    output_dim = 1
    input_dim = 20

    time_series = time_series_whole[begin:sample_interval:end]
    x_data = time_series[1:(end-output_dim)]
    y_data = time_series[(1+input_dim):end]
    x_data_win = zeros((size(x_data, 1) - input_dim + 1), input_dim)

    for j in 1:(size(x_data, 1)-input_dim+1)
        x_data_win[j, :] = x_data[j:(j+input_dim-1)]
    end



    x_data_vel = zeros((size(x_data, 1) - input_dim + 1), 2)
    y_data_vel = zeros((size(x_data, 1) - input_dim + 1), 2)

    x_data_vel[:, 1] .= x_data_win[:, end]
    y_data_vel[:, 1] .= y_data
    for i in 1:size(x_data_win, 1)
        v_arr = []
        for j in 1:(size(x_data_win, 2)-1)
            v_i = (x_data_win[i, j] - x_data_win[i, j+1]) / dt
            push!(v_arr, v_i)
        end
        v_avg = mean(v_arr)
        x_data_vel[i, 2] = v_avg
    end
    y_data_vel[1:end-1, 2] .= x_data_vel[2:end, 2]

    return time_series[1:end-1], x_data_win[1:end-1, :], x_data_vel[1:end-1, :], y_data[1:end-1], y_data_vel[1:end-1, :]
end

function save_data(x_train, x_test, y_train, y_test; kws...)
    args = Args_data(; kws...)
    @save "../data/encoded_time_series/$(args.variation)_variation/x_train_$(args.data_set).bson" x_train
    @save "../data/encoded_time_series/$(args.variation)_variation/x_test_$(args.data_set).bson" x_test
    @save "../data/encoded_time_series/$(args.variation)_variation/y_train_$(args.data_set).bson" y_train
    @save "../data/encoded_time_series/$(args.variation)_variation/y_test_$(args.data_set).bson" y_test
end

function load_video(n; kws...)
    args = Args_data(; kws...)
    video = VideoIO.load("../data/video/time_series/$(args.variation)_variation/$(args.data_set)_$(n).mp4")
    return video
end

# --- Load Encoder --- #

encoder_μ = load_encoder()

# --- Load Data --- #
video1 = load_video(1)
video2 = load_video(2)
video3 = load_video(3)
video4 = load_video(4)

# --- Encode video --- #
time_series1 = video2time_series(video1, encoder_μ)
time_series2 = video2time_series(video2, encoder_μ)
time_series3 = video2time_series(video3, encoder_μ)
time_series4 = video2time_series(video4, encoder_μ)


time_series_1, x_data_win1, x_data_vel1, y_data1, y_data_vel1 = time_series1
time_series_2, x_data_win2, x_data_vel2, y_data2, y_data_vel2 = time_series2
time_series_3, x_data_win3, x_data_vel3, y_data3, y_data_vel3 = time_series3
time_series_4, x_data_win4, x_data_vel4, y_data4, y_data_vel4 = time_series4


x_train = permutedims(vcat(x_data_win1, x_data_win2, x_data_win3), [2, 1])
x_test = x_data_win4

y_train = unsqueeze(vcat(y_data1, y_data2, y_data3), 1)
y_test = unsqueeze(y_data4, 1)

save_data(x_train, x_test, y_train, y_test)



