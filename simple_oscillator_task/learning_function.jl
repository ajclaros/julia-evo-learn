include("./ctrnn.jl")
include("./rl_ctrnn.jl")
include("./learning_rule.jl")
include("./fitness_function.jl")

function CPG(system, ;size::Int64)
    system[1].inputs .= zeros(size)
end

function prelearningCPG(system,;size::Int64, dt::Float64)
    CPG(system,  size=size)
    stepRL(system, dt)
    reward = reward_function(system[1], system[3],  false)
end

function learningCPG(system, ;size::Int64, dt::Float64)
    CPG(system,   size=size)
    stepRL(system, 0.1)
    iter_moment(dt, system[2][1], system[2][2])
    reward = reward_function(system[1], system[3],  true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
end

function learn(
    genome,
    ;params_duration = 5000,
    params_size = 3,
    params_window_size = 200,
    params_learn_rate = 0.05,
    params_conv_rate = 0.0001,
    params_init_flux = 1.0,
    params_max_flux = 10.0,
    params_period_min = 100,
    params_period_max = 300,
    params_learning_start = 200,
    params_dt = 0.01
    )
    ns = createRLCTRNN(genome, params_size, params_learn_rate, params_conv_rate,
                       params_window_size, 0.0, params_period_min, params_period_max,
                       params_dt, 0.0, 0.0, -16.0, 16.0, params_init_flux, params_max_flux,
                       16.0,16.0, 5.0, 6.0
                       )
    start_genome = recoverParameters(ns[1])
    duration = Int(params_duration / params_dt)
    pre_learning = range(1, stop=params_learning_start,step=params_dt)
    during_learning = range(start=params_learning_start, stop=params_duration, step=params_dt)
    for i in 1:length(pre_learning)
        prelearningCPG(ns, size=params_size, dt=params_dt)
    end
    for i in 0:length(during_learning)
        learningCPG(ns, size=params_size, dt=params_dt)
    end
    start_fitness = fitness_function_oscillate(start_genome, params_size)
    end_genome = recoverParameters(ns[1], extended=false)
    end_fitness = fitness_function_oscillate(end_genome , params_size)
    return end_fitness
end
# genome = npzread("./evolved-osc/2/fit-20034.npy")
# x = learn(genome)
# println("End fitness: ", x)
