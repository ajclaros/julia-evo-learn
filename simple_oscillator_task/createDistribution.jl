# Runs learning algorithm N times and plots the end performance as a histogram
using Colors
using PyPlot
using Distributed
using JLD2
@everywhere begin
    using NPZ
    include("./learning_function.jl")

    function get_indices(a, b, fitness)
        return findall(x -> x >= a && x <= b, fitness)
    end

    function get_fitness(filename)
        println("Loading $filename")
        return parse(Int, split(filename, "-")[2][1:5])/100000
    end

    function drop_outliers!(fitness, min, max, string_list)
        for i in length(fitness):-1:1
            if fitness[i] < min || fitness[i] > max
                deleteat!(fitness, i)
                deleteat!(string_list, i)
            end
        end
    end

    function getPathname(params_size::Int64)
        return "./evolved-osc/" * string(params_size) * "/"
    end

    function parallelLearning(genome, genome_file, ix;
                              params_duration::Int64 = 20000,
                              params_size::Int64 = 3,
                              params_window_size::Int64 = 220,
                              params_learn_rate::Float64 = 0.1,
                              params_conv_rate::Float64 = 0.000000001,
                              params_init_flux::Float64 = 0.02,
                              params_max_flux::Float64 = 0.2,
                              params_period_min::Int64 = 440,
                              params_period_max::Int64 = 4400,
                              params_learning_start::Int64 = 3000,
                              params_dt = 0.1,
                              params_num_trials::Int64 = 1,
                              )
        println("\nGenome: $(ix) ", genome_file)
        fitness = zeros(params_num_trials)
        Threads.@threads for i in 1:params_num_trials
            fitness[i] = learn(genome,
                               params_duration=params_duration,
                               params_size=params_size,
                               params_window_size=params_window_size,
                               params_learn_rate=params_learn_rate,
                               params_conv_rate=params_conv_rate,
                               params_init_flux=params_init_flux,
                               params_max_flux=params_max_flux,
                               params_period_min=params_period_min,
                               params_period_max=params_period_max,
                               params_learning_start=params_learning_start,
                               params_dt=params_dt,
                               )
        end
        return genome_file, fitness
    end
end

# function main() # uncomment to run as script, comment out to run in REPL
    p = Dict(
        :duration => 2000,
        :size => 2,
        :window_size => 200,
        :learn_rate => 0.05,
        :conv_rate => 0.0001,
        :init_flux => 1.0,
        :max_flux => 05.0,
        :period_min => 100,
        :period_max => 300,
        :learning_start => 300,
        :dt => 0.1,
        :num_trials => 100,
        :fit_range => (0.20, 0.25),
        :choose_random => true,
        :num_random => 20,
    )

    pathname = getPathname(p[:size])
    files = readdir(pathname)

    # filter out files that are not npz
    files = filter(x -> endswith(x, ".npy"), files)
    fitness = [get_fitness(file) for file in files]
    println("Number of files: ", length(files))
    drop_outliers!(fitness, p[:fit_range][1], p[:fit_range][2], files)
    if p[:choose_random]
        files = files[rand(1:length(files), p[:num_random])]
        fitness = [get_fitness(file) for file in files]
    end

    results_dict = Dict(file => zeros(p[:num_trials]) for file in files)
    # reorder results_dict to be in order of starting fitness
    results_dict = Dict(sort(collect(results_dict), by=x->get_fitness(x[1]))...)


    println("Number reduced to: ", length(files))
    #distributed and multithreaded
    results = pmap(i -> parallelLearning(
        npzread(pathname * files[i]), files[i], i;
        params_duration=p[:duration],
        params_size=p[:size],
        params_window_size=p[:window_size],
        params_learn_rate=p[:learn_rate],
        params_conv_rate=p[:conv_rate],
        params_init_flux=p[:init_flux],
        params_max_flux=p[:max_flux],
        params_period_min=p[:period_min],
        params_period_max=p[:period_max],
        params_learning_start=p[:learning_start],
        params_dt=p[:dt],
        params_num_trials=p[:num_trials],
        ), 1:length(files))
    for (file, fitness) in results
        results_dict[file] = fitness
    end



    # create a histogram for each file

    x_0 = get_fitness(collect(keys(results_dict))[1])
    cols = distinguishable_colors(length(files), [RGB(0,0,0), RGB(1,1,1)], dropseed=true)
    pcols = map(col -> (red(col), green(col), blue(col)), cols)
    fig, ax = subplots(nrows=Int(ceil(length(files)/2)), ncols=2, figsize=(10, 5))
    for ix in 1:length(results)
        file = files[ix]
        genome = npzread(pathname * file)
        starting_fitness = fitness_function_oscillate(genome, p[:size], 220.0)
        value = results_dict[file]
        arr = vcat(value, [starting_fitness])
        subplots_adjust(hspace=0.70)
        # plt.hist(value, bins=20, color=pcols[ix], alpha=0.5, label="Learned Fitness")
        ax[ix].hist(value, bins=20, color=pcols[ix])
        if ix==1
            ax[ix].axvline(starting_fitness, color="k", linestyle="dashed", linewidth=2, label="Starting Fitness")
            ax[ix].axvline(mean(value), color="r", linestyle="dashed", linewidth=2, label="Mean")
        else
            ax[ix].axvline(starting_fitness, color="k", linestyle="dashed", linewidth=2)
            ax[ix].axvline(mean(value), color="r", linestyle="dashed", linewidth=2)
        end
        # set xlims
        max_val = maximum(arr)
        # round up to  nearest 0.05
        max_val = ceil(max_val * 20) / 20
        # max_val = round(max_val, digits=2)
        ax[ix].set_xlim(x_0 , max_val)
        # xlims = ax[ix].get_xlim()
        # if xlims[1] < 0.0
        #     ax[ix].axes[:set_xticks]([0, round(xlims[2] ;digits=3)])
        # else
        #     ax[ix].axes[:set_xticks]([round(xlims[1] ; digits=3), round(xlims[2] ;digits=3)])
        # end
        # ax[ix].axes[:set_xticks](max_val)
        ax[ix].axes[:set_yticks]([])
        # set y axis title
        ax[ix].set_ylabel("($(ix))  ")
        # rotate y axis title
        ax[ix].yaxis.label.set_rotation(0)
    end
    fig.legend(loc="upper center")
    fig.suptitle("Fitness Distribution with Duration $(p[:duration])")
    savefig("./images/oscillation_random-$(p[:num_random])genomes-home")
# end # end main, uncomment to run as a script. keep commented to run in REPL
# save results_dict as a .jld2 file
JLD2.@save "results_dict_duration1000.jld2" results_dict
# get fitness of the first key in results_dict
