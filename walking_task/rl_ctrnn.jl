include("./ctrnn.jl")
include("./learning_rule.jl")
# import plotting library

mutable struct RLCTRNN
    n::Int64
    time_constants::Vector{Float64}
    inv_time_constants::Vector{Float64}
    inner_biases::Vector{Float64}
    extended_biases::Vector{Float64}
    inner_weights::Matrix{Float64}
    extended_weights::Matrix{Float64}
    inputs::Vector{Float64}
    voltages::Vector{Float64}
    outputs::Vector{Float64}
    reward::Float64
    netinput::Vector{Float64}
end

mutable struct RollingWindows
    window_size::Int64
    large_window_size::Int64
    distance::Vector{Float64}
    window_a::Vector{Float64}
    window_b::Vector{Float64}
    delay::Int64
    dt::Float64
    index::Int64
    windowed::Int64
    windowed2::Int64
    window_index::Int64
    distance_index::Int64
    delayed_index::Int64
end


RLCTRNN(n::Int) = RLCTRNN(n,
                          zeros(n), zeros(n),
                          zeros(n), zeros(n),
                          zeros(n,n), zeros(n,n),
                          zeros(n), zeros(n), zeros(n),
                           0.0, zeros(n))

RollingWindows(window_size::Int, delay, dt) = RollingWindows(window_size,
                                                             Int(window_size + delay),
                                                             zeros(Int(window_size/dt + delay/dt+100)),
                                                             zeros(Int(window_size/dt)),
                                                             zeros(Int(window_size/dt)),
                                                             Int(delay/dt), dt,
                                                                1, 1, 1, 1, 1, 1)




function createRLCTRNN(genome, n, learn_rate, conv_rate, window_size, delay,
                       period_min, period_max, dt,
                       performance_bias, tolerance,
                       p_min, p_max, init_flux, max_flux,
                       WR, BR, TR, TA,
                       )
    center_mat = zeros(n+1, n)
    ns = createNervousSystem(genome, n)
    ns_nodes = ns[1]
    ns_states = ns[2]
    for i in 1:n
        for j in 1:n
            center_mat[i,j] = ns_nodes.weights[i,j]
        end
        center_mat[n+1, i] = ns_nodes.biases[i]
    end
    learning_system = createLRuleHomogenous(learn_rate, conv_rate,
                                            performance_bias, tolerance,
                                            p_min, p_max,
                                            period_min, period_max,
                                            init_flux, max_flux,
                                            center_mat)
    learner_params = learning_system[1]
    learner_matrices = learning_system[2]
    rlctrnn = RLCTRNN(n)
    rlctrnn.inner_weights = view(learner_matrices.center_mat, 1:n, 1:n)
    rlctrnn.extended_weights = view(learner_matrices.extended_mat, 1:n, 1:n)
    rlctrnn.inner_biases = view(learner_matrices.center_mat, n+1, 1:n)
    rlctrnn.extended_biases = view(learner_matrices.extended_mat, n+1, 1:n)
    rlctrnn.time_constants = ns_nodes.time_constants
    rlctrnn.inv_time_constants = ns_nodes.inv_time_constants
    rolling_windows = RollingWindows(window_size, delay, dt)
    return (rlctrnn, learning_system, rolling_windows)
end
createRLCTRNN(genome, n, learn_rate, conv_rate, window_size, delay) = createRLCTRNN(
    genome, n, learn_rate, conv_rate, window_size, delay,
    window_size, 10*window_size, 0.1,
    0.0, 0.0,
    -16.0, 16.0, 1.0, 5.0,
    16.0, 16.0, 5.0, 6.0)
createRLCTRNN(genome, n, learn_rate, conv_rate) = createRLCTRNN(
    genome, n, learn_rate, conv_rate, 440, 310,
    440, 10*440, 0.1,
    0.0, 0.0,
    -16.0, 16.0, 1.0, 5.0,
    16.0, 16.0, 5.0, 6.0)

# create a function that iterates through rlctrnn and plots the inner weights in black and the extended weights in blue

function simulate(system, duration, dt, record_every)
    rlctrnn = system[1]
    learning_system = system[2]
    rolling_windows = system[3]
    learning_params = learning_system[1]
    learning_matrices = learning_system[2]
    sim_time = range(0, stop=duration, step=dt)
    record_arr = Int(floor(duration/dt/record_every))
    println("Size of record array: ", record_arr)
    println(length(sim_time))
    println(record_arr)
    inner_weights = zeros(3,3, record_arr)
    extended_weights = zeros(3,3,record_arr)
    for i in 0:length(sim_time)
        iter_moment(dt , learning_matrices, learning_params)
        update_weights_with_reward(rand()-0.5 , learning_matrices, learning_params)
        if i%record_every == 0
            inner_weights[:,:,Int(i/record_every)] = rlctrnn.inner_weights
            extended_weights[:,:,Int(i/record_every)] = rlctrnn.extended_weights
        end
    end
    return inner_weights, extended_weights
end

function reward_function(rlctrnn, windows, distance, learning)
    windows.delayed_index = mod1((windows.index + windows.delay), length(windows.distance))
    windows.windowed = windows.index - length(windows.window_b)
    windows.windowed2 = windows.index - 2*length(windows.window_b)
    windows.window_index = mod1(windows.index,length(windows.window_b))
    windows.distance_index = mod1(windows.index, length(windows.distance))
    update_windows(distance, windows)
    windows.index +=1
    if !learning
        return 0
    end
    rlctrnn.reward = mean(windows.window_b) - mean(windows.window_a)
end

function reward_function(rlctrnn, windows, distance, learning)
    windows.delayed_index = mod1((windows.index + windows.delay), length(windows.distance))
    windows.windowed = windows.index - length(windows.window_b)
    windows.windowed2 = windows.index - 2*length(windows.window_b)
    windows.window_index = mod1(windows.index,length(windows.window_b))
    windows.distance_index = mod1(windows.index, length(windows.distance))
    update_windows(distance, windows)
    windows.index +=1
    if !learning
        return 0
    end
    rlctrnn.reward = mean(windows.window_b) - mean(windows.window_a)
end
function update_windows(distance, windows)
    windows.distance[windows.delayed_index] = distance
    windows.window_b[windows.window_index] = (windows.distance[mod1(windows.distance_index, length(windows.distance))]
                                                - windows.distance[mod1(windows.windowed , length(windows.distance))])/(windows.window_size)
    windows.window_a[windows.window_index] = windows.window_b[mod1(windows.window_index+1, length(windows.window_b))]

end

function stepRL(system, dt::Float64)
    n = system[1].n
    rlctrnn = system[1]
    learning_system = system[2]
    rlctrnn.inner_weights = view(learning_system[2].center_mat, 1:n, 1:n)
    rlctrnn.inner_biases = view(learning_system[2].center_mat, n+1, 1:n)
    rlctrnn.extended_weights = view(learning_system[2].extended_mat, 1:n, 1:n)
    rlctrnn.extended_biases = view(learning_system[2].extended_mat, n+1, 1:n)
    rlctrnn.netinput .= rlctrnn.inputs .+ dotProductRowColumn(transpose(rlctrnn.extended_weights), rlctrnn.outputs)
    rlctrnn.voltages .+= dt .*(rlctrnn.inv_time_constants .* (-rlctrnn.voltages .+ rlctrnn.netinput))
    rlctrnn.outputs .= sigmoid.(rlctrnn.voltages .+ rlctrnn.extended_biases)
end

function step(system, dt::Float64)
    n = system[1].n
    rlctrnn = system[1]
    learning_system = system[2]
    rlctrnn.inner_weights = view(learning_system[2].center_mat, 1:n, 1:n)
    rlctrnn.inner_biases = view(learning_system[2].center_mat, n+1, 1:n)
    rlctrnn.netinput .= rlctrnn.inputs .+ dotProductRowColumn(transpose(rlctrnn.inner_weights) , rlctrnn.outputs)
    rlctrnn.voltages .+= dt .* (rlctrnn.inv_time_constants .* (-rlctrnn.voltages .+ rlctrnn.netinput))
    rlctrnn.outputs .= sigmoid.(rlctrnn.voltages .+ rlctrnn.inner_biases)
end

# system = createRLCTRNN(testgenome, 3, 0.1, 0.1, 440, 310, 440, 10*440, 0.1, 0.0, 0.0, -16.0, 16.0, 1.0, 5.0, 16.0, 16.0, 5.0, 6.0)
#  rlctrnn = system[1]
# using BenchmarkTools
# @btime step(rlctrnn, 0.1)
# @btime stepRL(rlctrnn, 0.1)
# @code_warntype stepRL(rlctrnn, 0.1)

function recoverParameters(ns::RLCTRNN, ;extended=true)
    if extended
        vcat(recoverWeight(ns.extended_weights, ns.n), recoverBias(ns.extended_biases), recoverTimeConstant(ns.time_constants))
    else
        vcat(recoverWeight(ns.inner_weights, ns.n), recoverBias(ns.inner_biases), recoverTimeConstant(ns.time_constants))
    end
end
