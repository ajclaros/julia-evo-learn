include("./ctrnn.jl")
include("./leggedwalker.jl")
using NPZ

function createAgent(genome::Vector{Float64}, N::Int64)
    # Create a CTRNN with N neurons
    ctrnn = createNervousSystem(genome, N)
    leggedAgent = createLeggedAgent()

    return (ctrnn, leggedAgent)
end

"Fitness function for a single legged walker driven by a CTRNN.
Fitness is calculated by the distance travelled over a duration of time.
Verified to output the same as the Python version. and is 1.6x faster."
function fitness_function(genome::Array{Float64}, N::Int64,
                          generator_type::String, configuration::Int64, duration::Float64,
                          dt::Float64)
    # Create a legged walker with the given CTRNN
    agent = createAgent(genome, N)
    time = range(0, stop=duration-dt, step=dt)
    ns = agent[1]
    ns_nodes = ns[1]
    ns_state = ns[2]
    walker = agent[2]

    for i in 1:length(time)
        if generator_type == "RPG"
            ns_state.inputs = ones(N) .* getAngleFeedback(walker)
        else
            ns_state.inputs = zeros(N)
        end
        ctrnnStep(dt, ns_nodes, ns_state)
        stepNWalker(walker, dt, configuration, ns_state.outputs)
    end
    return walker[2].cx / duration
end
fitness_function(genome::Vector{Float64}, N::Int) = fitness_function(genome, N,  "RPG", 1, 220.0, 0.1)
fitness_function(genome, N, generator_type) = fitness_function(genome, N, generator_type, 1, 220.0, 0.1)
fitness_function(genome, N, generator_type, configuration) = fitness_function(genome, N, generator_type, configuration, 220.0, 0.1)
fitness_function(genome, N, generator_type, configuration, duration) = fitness_function(genome, N, generator_type, configuration, duration, 0.1)


function get_fitness(filename)
    return parse(Int, split(filename, "-")[2][1:5])/100000
end

function get_fitness(filename)
    return parse(Int, split(filename, "-")[2][1:5])/100000
end

function sort_string_based_on_numerical_list(string_list, num_list)
    sorted_list = sortperm(num_list)
    return string_list[sorted_list]
end

"function to compare numpy filename with fitness in the julia implementation"
function fitness_main()
    trial_params = Dict(
        :size => 3,
        :generator => "CPG",
        :config  => "012",
    )
    pathname = "./evolved/$(trial_params[:generator])/$(trial_params[:size])/$(trial_params[:config])/"

    x = readdir(pathname)
    fitnesses = [get_fitness(filename) for filename in x]
    sorted_filenames = sort_string_based_on_numerical_list(x, fitnesses)
    sorted_filenames = sorted_filenames[1:2]
    print(sorted_filenames)
    config = 1
    if trial_params[:config] =="0"
        config = 1
    elseif trial_params[:config] =="01"
        config = 2
    elseif trial_params[:config] =="012"
        config = 3
    end
    for filename in sorted_filenames
        genome = npzread(pathname * filename)
        fitness = fitness_function(genome, trial_params[:size], trial_params[:generator], config)
        fit_diff = fitness - get_fitness(filename)
        println("-----")
        println(fitness)
        println(get_fitness(filename))
        if fit_diff>1e-5
            println("Difference from file and measured difference: $(fitness- get_fitness(filename))")
        end
    end
end

function fitness_function_oscillate(genome::Array{Float64}, N::Int64, dt::Float64, duration::Float64)
    ns = createNervousSystem(genome, N)
    ns_nodes = ns[1]
    ns_state = ns[2]
    time = range(0, stop=duration-dt, step=dt)
    change_in_outputs = 0.0
    past_outputs = zeros(N)
    for i in 1:length(time)
        past_outputs = copy(ns_state.outputs)
        ctrnnStep(dt, ns_nodes, ns_state)
        change_in_outputs += sum(abs.(ns_state.outputs .- past_outputs))
        # println(ns_state.inputs .* dotProductRowColumn(transpose(ns_nodes.weights), ns_state.outputs))

        
    end
    return change_in_outputs/N/duration
end
fitness_function_oscillate(genome::Vector{Float64}, N::Int) = fitness_function_oscillate(genome, N, 0.1, 220.0)
fitness_function_oscillate(genome::Vector{Float64}, N::Int, duration::Float64) = fitness_function_oscillate(genome, N, 0.1, duration)
