include("./ctrnn.jl")
include("./rl_ctrnn.jl")
include("./learning_rule.jl")
include("./fitness_function.jl")

function CPG(system, ;size::Int64)
    system[1].inputs .= zeros(size)
end

function RPG(system, body, ;size::Int64)
    system[1].inputs .= ones(size) .* getAngleFeedback(body)
end

function prelearningCPG(system, body, config, ;size::Int64, dt::Float64)
    CPG(system,  size=size)
    stepRL(system, dt)
    reward = reward_function(system[1], system[3], body[2].cx, false)
    stepNWalker(body, dt, config, system[1].outputs)
end

function prelearningRPG(system,  body, config; size::Int64, dt::Float64)
    RPG(system, body, size=size)
    step(system, dt)
    reward = reward_function(system[1], system[3], body[2].cx, false)
    stepNWalker(body, dt, config, system[1].outputs)
end
function learningCPG(system, body, config,;size::Int64, dt::Float64, tolerance::Float64)
    CPG(system,   size=size)
    stepRL(system, 0.1)
    iter_moment(dt, system[2][1], system[2][2])
    reward = reward_function(system[1], system[3], body[2].cx, true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
    stepNWalker(body, dt, config, system[1].outputs)
end

function learningRPG(system,  body, config,;size::Int64, dt::Float64, tolerance::Float64)
    RPG(system, body, size=size)
    stepRL(system, 0.1)
    iter_moment(dt, system[2][1], system[2][2])
    reward = reward_function(system[1], system[3], body[2].cx, true)
    update_weights_with_reward(reward, system[2][1], system[2][2])
    stepNWalker(body, dt, config, system[1].outputs)
end

function learn(
    genome,
    ;params_duration = 20000,
    params_size = 3,
    params_generator = "CPG",
    params_config = 3,
    params_window_size = 220,
    params_learn_rate = 0.1,
    params_conv_rate = 0.000000001,
    params_init_flux = 0.02,
    params_max_flux = 0.2,
    params_period_min = 440,
    params_period_max = 4400,
    params_learning_start = 3000,
    params_tolerance = 0.0,
    params_performance_bias = 0.0,
    params_dt = 0.1
    )
    body = createLeggedAgent()
    ns = createRLCTRNN(genome, params_size, params_learn_rate, params_conv_rate,
                       params_window_size, 0., params_period_min, params_period_max,
                       params_dt, 0.0, 0.0, -16.0, 16.0, params_init_flux, params_max_flux,
                       16.0,16.0, 5.0, 6.0
                       )
    start_genome = recoverParameters(ns[1])
    duration = Int(params_duration / params_dt)
    pre_learning = range(1, stop=params_learning_start,step=params_dt)
    during_learning = range(start=params_learning_start, stop=params_duration, step=params_dt)
    if params_generator == "CPG"
        for i in 1:length(pre_learning)
            prelearningCPG(ns, body, params_config, size=params_size, dt=params_dt)
        end
        for i in 0:length(during_learning)
            learningCPG(ns, body, params_config, size=params_size, dt=params_dt, tolerance=params_tolerance)
        end
    else
        for i in 1:length(pre_learning)
            prelearningRPG(ns, body, params_config, size=params_size, dt=params_dt)
        end
        for i in 0:length(during_learning)
            learningRPG(ns, body, params_config, size=params_size, dt=params_dt, tolerance=params_tolerance)
        end
    end

    # start_fitness = fitness_function(start_genome, params_size)
    # start_fitness = fitness_function(start_genome , params_size, params_generator, params_config)
    end_genome = recoverParameters(ns[1], extended=false)
    end_fitness = fitness_function(end_genome , params_size, params_generator, params_config)
    return end_fitness
end
# genome = npzread("./evolved/CPG/3/012/fit-55698.npy")
# x = learn(genome)
# println("End fitness: ", x)
