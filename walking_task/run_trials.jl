include("./ctrnn.jl")
include("./fitness_function.jl")
include("./leggedwalker.jl")
include("./rl_ctrnn.jl")
include("./learning_rule.jl")

using NPZ
using Colors
using Random
using PyPlot

function get_indices(a, b, fitness)
    return findall(x -> x >= a && x <= b, fitness)
end
function get_fitness(filename)
    return parse(Int, split(filename, "-")[2][1:5])/100000
end


# function display_plot(outputs, params)
#     tim = range(1, stop=params[:duration], step=params[:dt]*params[:record_every])
#     plot(time, outputs)
# end

#given a list of numbers, drop all values that are not within tuple range (min, max)
function drop_outliers!(fitness, min, max, string_list)
    for i in length(fitness):-1:1
        if fitness[i] < min || fitness[i] > max
            deleteat!(fitness, i)
            deleteat!(string_list, i)
        end
    end
end

function RPG(system, params::Dict{Any, Union{Any, Missing}}, body)
    system[1].inputs .= ones(params[:size]) .* getAngleFeedback(body)
end
function RPG(system, body, ;size::Int64)
    system[1].inputs .= ones(size) .* getAngleFeedback(body)
end

function CPG(system, params::Dict{Any, Union{Any, Missing}}, body)
    system[1].inputs .= zeros(params[:size])
end
function CPG(system, body, ;size::Int64)
    system[1].inputs .= zeros(size)
end
function prelearningRPG(system, params::Dict{Any, Union{Any, Missing}}, body, config)
    RPG(system, params, body)
    step(system, params[:dt])
    reward = reward_function(system[1], system[3], body[2].cx, false)
    stepNWalker(body, params[:dt], config, system[1].outputs)
end
function prelearningRPG(system,  body, config; size::Int64, dt::Float64)
    RPG(system, body, size=size)
    step(system, dt)
    reward = reward_function(system[1], system[3], body[2].cx, false)

    stepNWalker(body, dt, config, system[1].outputs)
end
function learningRPG(system, params::Dict{Any, Union{Any, Missing}}, body, config)
    RPG(system, params, body)
    stepRL(system, 0.1)
    iter_moment(params[:dt], system[2][1], system[2][2])
    reward = reward_function(system[1], system[3], body[2].cx, true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
    stepNWalker(body, params[:dt], config, system[1].outputs)
end

function learningRPG(system,  body, config,;size::Int64, dt::Float64, tolerance::Float64)
    RPG(system, body, size=size)
    stepRL(system, 0.1)
    iter_moment(dt, system[2][1], system[2][2])
    reward = reward_function(system[1], system[3], body[2].cx, true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
    stepNWalker(body, dt, config, system[1].outputs)
end
function prelearningCPG(system, params::Dict{Any, Union{Any, Missing}}, body, config)
    CPG(system, params, body)
    step(system, params[:dt])
    reward = reward_function(system[1], system[3], body[2].cx, false)
    stepNWalker(body, params[:dt], config, system[1].outputs)
end

function prelearningCPG(system,  body, config, ;size::Int64, dt::Float64)
    CPG(system, body, size=size)
    step(system, dt)
    reward = reward_function(system[1], system[3], body[2].cx, false)
    stepNWalker(body, dt, config, system[1].outputs)
end

function learningCPG(system, params::Dict{Any, Union{Any, Missing}}, body, config)
    CPG(system, params, body)
    stepRL(system, 0.1)
    iter_moment(params[:dt], system[2][1], system[2][2])
    reward = reward_function(system[1], system[3], body[2].cx, true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
    stepNWalker(body, params[:dt], config, system[1].outputs)
end

function learningCPG(system, body, config, ;size::Int64, dt::Float64, tolerance::Float64)
    CPG(system,  body, size=size)
    stepRL(system, 0.1)
    iter_moment(dt, system[2][1], system[2][2])
    reward = reward_function(system[1], system[3], body[2].cx, true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
    stepNWalker(body, dt, config, system[1].outputs)
end
function run_trial(
    index,
    ;params_duration=10000,
    params_dt=0.1,
    params_record_every=10,
    params_size=3,
    params_delay=330,
    params_generator = "CPG",
    params_config = "012",
    params_window_size = 220,
    params_learn_rate = 0.01,
    params_conv_rate = 0.000000001,
    params_init_flux = 0.20,
    params_max_flux = 0.20,
    params_period_min = 440,
    params_period_max = 4400,
    params_learning_start = 3000,
    params_tolerance = 0.0,
    params_performance_bias = 0.0,
    params_fit_range = (0.2, 0.7),
    )
    pathname = "./evolved/$(params_generator)/$(params_size)/$(params_config)/"
    x = readdir(pathname)
    fitness = [get_fitness(filename) for filename in x]
    sorted_fitness = sort(fitness)
    sorted_filenames = sort_string_based_on_numerical_list(x, fitness)
    drop_outliers!(sorted_fitness, params_fit_range[1], params_fit_range[2], sorted_filenames)
    # get indices of all elements in fitness that are within the range [a, b]
    genome = npzread(pathname * sorted_filenames[index])
    # create a legged walker
    if params_config == "0"
        config = 1
    elseif params_config == "01"
        config = 2
    elseif params_config == "012"
        config = 3
    end
    body = createLeggedAgent()
    ns = createRLCTRNN(genome, params_size, params_learn_rate, params_conv_rate,
                       params_window_size, params_delay, params_period_min, params_period_max,
                       params_dt, params_performance_bias, params_tolerance,
                       -16.0, 16.0, params_init_flux, params_max_flux,
                       16.0, 16.0, 5.0, 6.0)
    average_performance = zeros(Int(params_duration/params_dt))
    record_arr_length = Int(params_duration/params_dt/params_record_every)
    inner_weights = zeros(record_arr_length)
    extended_weights = zeros(record_arr_length)
    window_b_track = zeros(record_arr_length)
    window_a_track = zeros(record_arr_length)
    reward_track= zeros(record_arr_length)
    flux_track= zeros(record_arr_length)
    duration = Int(params_duration/params_dt)
    pre_learning = range(1, stop=params_learning_start, step=params_dt*params_record_every)
    during_learning = range(start=params_learning_start, stop=params_duration, step=params_dt*params_record_every)
    learning_start_ix = Int(params_learning_start/params_dt/params_record_every)
    t = 0
    if params_generator=="RPG"
        for i in 1:length(pre_learning)
            for j in 1:params_record_every
                prelearningRPG(ns, body, config, size=params_size, dt=params_dt)
            end
            inner_weights[i] = ns[1].inner_weights[1,1]
            extended_weights[i] = ns[1].extended_weights[1,1]
            window_b_track[i] = mean(ns[3].window_b)
            window_a_track[i] = mean(ns[3].window_a)
            reward_track[i] = ns[1].reward
            flux_track[i] = ns[2][2].flux_mat[1,1]
        end
        for i in 0:length(during_learning)-1
            for j in 1:params_record_every
                learningRPG(ns, body, config, size=params_size, dt=params_dt, tolerance=params_tolerance)
            end
            window_b_track[learning_start_ix+i] = mean(ns[3].window_b)
            window_a_track[learning_start_ix+i] = mean(ns[3].window_a)
            inner_weights[learning_start_ix+i] = ns[1].inner_weights[1,1]
            extended_weights[learning_start_ix+i] = ns[1].extended_weights[1,1]
            reward_track[learning_start_ix+i] = ns[1].reward
            flux_track[learning_start_ix+i] = ns[2][2].flux_mat[1,1]
        end
    end
    if params_generator=="CPG"
        for i in 1:length(pre_learning)
            for j in 1:params_record_every
                prelearningCPG(ns, body, config, size=params_size, dt=params_dt)
            end
            window_b_track[i] = mean(ns[3].window_b)
            window_a_track[i] = mean(ns[3].window_a)
            inner_weights[i] = ns[1].inner_weights[1,1]
            extended_weights[i] = ns[1].extended_weights[1,1]
            reward_track[i] = ns[1].reward
            flux_track[i] = ns[2][2].flux_mat[1,1]
        end
        for i in 0:length(during_learning)-1
            for j in 1:params_record_every
                learningCPG(ns, body, config, size=params_size, dt=params_dt, tolerance=params_tolerance)
            end
            window_b_track[learning_start_ix+i] = mean(ns[3].window_b)
            window_a_track[learning_start_ix+i] = mean(ns[3].window_a)
            inner_weights[learning_start_ix+i] = ns[1].inner_weights[1,1]
            extended_weights[learning_start_ix+i] = ns[1].extended_weights[1,1]
            reward_track[learning_start_ix+i] = ns[1].reward
            flux_track[learning_start_ix+i] = ns[2][2].flux_mat[1,1]
        end
    end
    # display_plot(window_b_track, params)
    # TODO create recoverParameters function
    return Dict(:window_b => window_b_track, :window_a => window_a_track,
                :inner_weights => inner_weights, :extended_weights => extended_weights,
                :reward => reward_track, :flux => flux_track,
                :starting_fitness=>sorted_fitness[index],
                :end_fitness => mean(ns[3].window_b),
                :genome => genome,
                :dt => params_dt,
                :record_every => params_record_every,
                :duration => params_duration,
                :learning_start => params_learning_start,
                :size => params_size,
                :learn_rate => params_learn_rate,
                :conv_rate => params_conv_rate,
                :window_size => params_window_size,
                :delay => params_delay,
                :period_min => params_period_min,
                :period_max => params_period_max,
                :performance_bias => params_performance_bias,
                :tolerance => params_tolerance,
                :init_flux => params_init_flux,
                :max_flux => params_max_flux,
                :generator => params_generator,
                :config => params_config,
                :index => index)
end

p = Dict(
   :duration => 20000,
   :size => 3,
   :delay => 0,
   :generator => "CPG",
   :config  => "012",
   :window_size => 220,
   :learn_rate => 0.1,
   :conv_rate =>  0.000000001,
   :init_flux => 0.02,
   :max_flux  => 0.2,
   :period_min => 1000,
   :period_max => 10000,
   :learning_start => 3000,
   :tolerance => 0.000000,
   :performance_bias => 0.000,
   :fit_range => (0.50, 0.70),
   :dt => 0.1,
   :record_every => 10,
   :num_trials => 8,
   :indices => [19]
)
# enter values of trial params into run_trial
results = []
for index in p[:indices]
        Threads.@threads for j in 1:p[:num_trials]
            println("Running index: $index, trial: $j ")
            push!(results, run_trial(index,
                  ;params_duration = p[:duration],
                    params_size = p[:size],
                    params_delay = p[:delay],
                    params_generator = p[:generator],
                    params_config = p[:config],
                    params_window_size = p[:window_size],
                    params_learn_rate = p[:learn_rate],
                    params_conv_rate = p[:conv_rate],
                    params_init_flux = p[:init_flux],
                    params_max_flux = p[:max_flux],
                    params_period_min = p[:period_min],
                    params_period_max = p[:period_max],
                    params_learning_start = p[:learning_start],
                    params_tolerance = p[:tolerance],
                    params_performance_bias = p[:performance_bias],
                    params_fit_range = p[:fit_range],
                    params_dt = p[:dt],
                    params_record_every = p[:record_every],
             ))
     end
end
# from each result, plot the window_b track
function plot_results(results, p)
    cols = distinguishable_colors(length(p[:indices]), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    pcols = map(col -> (red(col), green(col), blue(col)), cols)
    subplot(2,1,1)
    subplots_adjust(hspace=0.15)
    for i in 1:length(results)
        ix = findall(x->x==results[i][:index], p[:indices])
        plot(results[i][:window_b], c=pcols[ix[1]])
    end
    PyPlot.title("Average performance: \n $(p[:generator]) size:$(p[:size]) config: $(p[:config])")
    subplot(2,1,2)
    for i in 1:length(results)
        ix = findall(x->x==results[i][:index], p[:indices])
        plot(results[i][:flux], c=pcols[ix[1]])
    end
    PyPlot.title("Flux size")
end
plot_results(results, p)
println("Done")
