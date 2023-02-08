# Runs learning algorithm N times and plots the end performance as a histogram
using Colors
using PyPlot
using Distributed
@everywhere begin
    using NPZ
    include("./learning_function.jl")

    function get_indices(a, b, fitness)
        return findall(x -> x >= a && x <= b, fitness)
    end
    function get_fitness(filename)
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
    function numerical_config(params_config::String)
        if params_config == "0"
            return 1
        elseif params_config == "01"
            return 2
        elseif params_config == "012"
            return 3
        end
    end

    function string_config(params_config::Int64)
        if params_config == 1
            return "0"
        elseif params_config == 2
            return "01"
        elseif params_config == 3
            return "012"
        end
    end
    function getPathname(params_generator::String, params_size::Int64, params_config::String)
        return "./evolved/" * params_generator * "/" * string(params_size) * "/" * params_config * "/"
    end

    function getPathname(params_generator::String, params_size::Int64, params_config::Int64)
        return getPathname(params_generator, params_size, string_config(params_config))
    end

    function parallelLearning(genome, genome_file;
                              params_duration::Int64 = 20000,
                              params_size::Int64 = 3,
                              params_generator::String = "CPG",
                              params_config::Int64 = 3,
                              params_window_size::Int64 = 220,
                              params_learn_rate::Float64 = 0.1,
                              params_conv_rate::Float64 = 0.000000001,
                              params_init_flux::Float64 = 0.02,
                              params_max_flux::Float64 = 0.2,
                              params_period_min::Int64 = 440,
                              params_period_max::Int64 = 4400,
                              params_learning_start::Int64 = 3000,
                              params_tolerance = 0.0,
                              params_performance_bias = 0.0,
                              params_dt = 0.1,
                              params_num_trials::Int64 = 1,
                              )
        println("Genome: ", genome_file)
        fitness = zeros(params_num_trials)
        Threads.@threads for i in 1:params_num_trials
            fitness[i] = learn(genome,
                               params_duration=params_duration,
                               params_size=params_size,
                               params_generator=params_generator,
                               params_config=params_config,
                               params_window_size=params_window_size,
                               params_learn_rate=params_learn_rate,
                               params_conv_rate=params_conv_rate,
                               params_init_flux=params_init_flux,
                               params_max_flux=params_max_flux,
                               params_period_min=params_period_min,
                               params_period_max=params_period_max,
                               params_learning_start=params_learning_start,
                               params_tolerance=params_tolerance,
                               params_performance_bias=params_performance_bias,
                               params_dt=params_dt,
                               )
        end
        return genome_file, fitness
    end
end


#function main() # uncomment to run as a script, comment to run in REPL
    p = Dict(
        :duration => 5000,
        :size => 3,
        :generator => "CPG",
        :config => "012",
        :window_size => 220,
        :learn_rate => 0.01,
        :conv_rate => 0.00000001,
        :init_flux => 0.02,
        :max_flux => 0.2,
        :period_min => 440,
        :period_max => 4400,
        :learning_start => 800,
        :tolerance => 0.0,
        :performance_bias => 0.0,
        :dt => 0.1,
        :num_trials => 200,
        :fit_range => (0.00, 0.70),
        :choose_random => true,
        :num_random => 20,
    )

    pathname = getPathname(p[:generator], p[:size], p[:config])
    files = readdir(pathname)
    fitness = [get_fitness(file) for file in files]
    println("Number of files: ", length(files))
    drop_outliers!(fitness, p[:fit_range][1], p[:fit_range][2], files)
    if p[:choose_random]
        files = files[rand(1:length(files), p[:num_random])]
        fitness = [get_fitness(file) for file in files]
    end
    results_dict = Dict(file => zeros(p[:num_trials]) for file in files)
    println("Number reduced to: ", length(files))
    config = numerical_config(p[:config])
    results = pmap(i -> parallelLearning(npzread(pathname * files[i]), files[i],
                                            params_duration = p[:duration],
                                            params_size = p[:size],
                                            params_generator = p[:generator],
                                            params_config = config,
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
                                            params_dt = p[:dt],
                                            params_num_trials = p[:num_trials]
                                            ), 1:length(files))
    for (file, fitness) in results
        results_dict[file] = fitness
    end
    # create a histogram for each file
    cols = distinguishable_colors(length(files), [RGB(0,0,0), RGB(1,1,1)], dropseed=true)
    pcols = map(col -> (red(col), green(col), blue(col)), cols)
    fig, ax = subplots(nrows=Int(ceil(length(files)/2)), ncols=2, figsize=(10, 5))
    for ix in 1:length(files)
        file = files[ix]
        genome = npzread(pathname * file)
        sconfig = numerical_config(p[:config])
        starting_fitness = fitness_function(genome, p[:size], p[:generator], sconfig, 220.0)
        value = results_dict[file]
        arr = vcat(value, [starting_fitness])
        subplots_adjust(hspace=0.70)
        ax[ix].hist(value, bins=20, color=pcols[ix])
        # find the minimum and maximum x values of the histogram
        xlims = ax[ix].get_xlim()
        if ix==1
            ax[ix].axvline(starting_fitness, color="k", linestyle="dashed", linewidth=2, label="Starting Fitness")
            ax[ix].axvline(mean(value), color="r", linestyle="dashed", linewidth=2, label="Mean")
        else
            ax[ix].axvline(starting_fitness, color="k", linestyle="dashed", linewidth=2)
            ax[ix].axvline(mean(value), color="r", linestyle="dashed", linewidth=2)
        end
        # ax[ix].axes[:set_xlim](min_value, max_value)
        # ax[i].xlim(min_value, max_value)
         # ax[ix].xticks([round(min_value; digits=3), round(max_value;digits=3)])
        # set x ticks to only be the min and max values rounded to 3 decimal places
        ax[ix].axes[:set_xticks]([round(xlims[1] ; digits=3), round(xlims[2] ;digits=3)])
        # plt.xlim(0.25, 0.50)
        # remove y ticks and values
        ax[ix].axes[:set_yticks]([])
    end
    # add legend to the figure at the top center
    fig.legend(loc="upper center")
    savefig("./images/walking_random-$(p[:num_random])-genomes")
    # fig.tight_layout()

    #position legend where the title would be

#end # end main, uncomment to run as a script keep commented to run in REPL

# @time main()
