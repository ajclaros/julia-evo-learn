# Runs learning algorithm N times and plots the end performance as a histogram
using Colors
# using PyPlot
using Distributed
using JLD2
using Random
using KernelDensityEstimate
using PyPlot
using PyCall
# using Seaborn

# plot using seaborn
@pyimport scipy.stats as stats
@pyimport seaborn as sns
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

    function drop_outliers(fitness, min, max, string_list)
        fitness = copy(fitness)
        string_list = copy(string_list)
        drop_outliers!(fitness, min, max, string_list)
        return string_list
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
    function getPathname(params_generator::String, params_size::Int64, params_config::Int64)
        return "./evolved/" * params_generator * "/" * string(params_size) * "/" * string_config(params_config) * "/"
    end

    function parallelLearning(genome, genome_file, ix;
                              params_duration::Int64 = 20000,
                              params_size::Int64 = 3,
                              params_generator::String = "CPG",
                              params_config::Int64 = 3,
                              params_window_size::Int64 = 220,
                              params_learn_rate::Float64 = 0.01,
                              params_conv_rate::Float64 = 0.00000001,
                              params_init_flux::Float64 = 0.02,
                              params_max_flux::Float64 = 0.2,
                              params_period_min::Int64 = 440,
                              params_period_max::Int64 = 4400,
                              params_learning_start::Int64 = 800,
                              params_dt = 0.1,
                              params_num_trials::Int64 = 1,
                              )
        println("Genome: $(ix) ", genome_file)
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
                               params_dt=params_dt,
                               )
        end
        return genome_file, fitness
    end
end

# function main() # uncomment to run as script, comment out to run in REPL

function get_results()
    p = Dict(
        :duration => 8000,
        :size => 3,
        :generator => "CPG",
        :config => 3,
        :window_size => 220,
        :learn_rate => 0.1,
        :conv_rate => 0.00000001,
        :init_flux => 0.02,
        :max_flux => 0.2,
        :period_min => 440,
        :period_max => 4400,
        :learning_start => 550,
        :dt => 0.1,
        :fitness_chunks => [0.16, 0.2, 0.3, 0.4, 0.5, 0.6], #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        :num_files_per_chunk => 5,
        :num_trials => 100,
        :choose_random => true,
    )

    pathname = getPathname(p[:generator], p[:size], p[:config])
    println("Pathname: ", pathname)
    files_arr = readdir(pathname)
    # filter out files that are not npz
    files_arr = filter(x -> endswith(x, ".npy"), files_arr)
    fitness = [get_fitness(file) for file in files_arr]
    println("Number of files: ", length(files_arr))
    results_dict = Dict()
    if p[:choose_random]
        for i in 1:length(p[:fitness_chunks])-1
            fit_min = p[:fitness_chunks][i]
            fit_max = p[:fitness_chunks][i+1]
            println("Fitness range: ", fit_min, " - ", fit_max)
            results_dict[fit_min] = Dict()
            indices = get_indices(fit_min, fit_max, fitness)
            files = drop_outliers(fitness[indices], fit_min, fit_max, files_arr[indices])
            # println("Chunk $i: $fit_min - $fit_max")
            # println("Number of files: ", length(indices))
            indices = 1:p[:num_files_per_chunk]
            if length(files) < p[:num_files_per_chunk]
                indices = 1:length(files)
            end
            println("Number of files: ", length(indices))
            results = pmap(ix -> parallelLearning(npzread(pathname * files[ix]),
                                                  files[ix],
                                                  ix,
                                                  params_duration=p[:duration],
                                                  params_size=p[:size],
                                                  params_generator=p[:generator],
                                                    params_config=p[:config],
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
                                                  ),
                           indices)
            for (file, fitness) in results
                results_dict[fit_min][file] = fitness
            end
        end
    end
    return results_dict, p
end
    # x_0 = get_fitness(keys[1])
function plot_results(results_dict, p)
    pathname = getPathname(p[:generator], p[:size], p[:config])
    cols = distinguishable_colors(p[:num_files_per_chunk], [RGB(0,0,0), RGB(1,1,1)], dropseed=true)
    pcols = map(col -> (red(col), green(col), blue(col)), cols)
    fig, ax = subplots(nrows=Int(ceil((length(p[:fitness_chunks]))/2)), ncols=2)
    # filenames = collect(keys(results_dict[0.1]))
    for l in 1:length(keys(results_dict))
        fit_val = collect(keys(results_dict))[l]
        filenames = collect(keys(results_dict[fit_val]))
        println("Plotting: ", filenames)
        results = zeros(length(filenames), p[:num_trials])
        println(fit_val)

        for i in 1:length(filenames)
            println(filenames[i])
            results[i, :] = results_dict[fit_val][filenames[i]]
        end
        starting_fitnesses = [get_fitness(file) for file in keys(results_dict[fit_val])]
        results = vec(results)
        results=  sort(results)
        min_val = minimum(starting_fitnesses)
        max_val = maximum(results)
        kernel = stats.gaussian_kde(results, bw_method=0.1)
        y_values = kernel(results)
        # scale y values to distribution
        y_values = y_values / maximum(y_values)
        subplots_adjust(hspace=0.4)
        # plot correct order of fit_val
        ix = findfirst(x -> x == fit_val, p[:fitness_chunks])
        # plot according by column then row
        if ix % 2 == 0
            col = 2
            row = Int(floor(ix/2))
        else
            col = 1
            row = Int(floor(ix/2)) + 1
        end
        ix = row + (col-1)*Int(ceil((length(p[:fitness_chunks])-2)/2))

        if ix == 1
            ax[ix].plot(results, y_values, color="k", label="Learned fitness KDE for closest genomes after starting point")
        else
            ax[ix].plot(results, y_values, color="k")
        end
        # sns.histplot(results, ax=ax[ix], alpha=1.0, color="k", label="Learned fitness KDE for $fit_val", stat="probability")
        ax[ix].fill_between(results, y_values, color="k", alpha=0.4)
        for i in 1:length(filenames)
            sns.histplot(
                data=results_dict[fit_val][filenames[i]], ax=ax[ix], color=pcols[i], alpha=0.5, stat="probability", binrange=(min_val, max_val), bins=20
            )
            genome = npzread(pathname * filenames[i])
            fitness = fitness_function(genome, p[:size], p[:generator], p[:config])
            if ix==1 && i==1
                ax[ix].axvline(fitness, color="k", label="Starting fitness", linestyle="--")
            else
                ax[ix].axvline(fitness, color=pcols[i], linestyle="--")
            end
        end
        buffer = (max_val - min_val) * 0.1
        ax[ix].axes.set_xlim([min_val-buffer, max_val+buffer])
        ax[ix].set_title("$(fit_val)")
        ax[ix].set_xlabel("Fitness")
        ax[ix].set_ylabel("Probability")
        fig.legend(loc="lower center", ncol=4)
        fig.suptitle("Distribution of fitness for 5 closest genomes after starting point\n Duration: $(p[:duration])")
    end
end
results_dict, p = get_results()
plot_results(results_dict, p)
# save results dict to file
# jldopen("./data/results_dict_test-dur$(p[:duration])-numg$(p[:num_files_per_chunk]).jld2", "w") do f
#     JLD2.write(f, "results_dict", results_dict)
#     JLD2.write(f, "params", p)
# end
# load results dict from file
# results_dict = jldopen("./data/results_dict_test-dur2000-numg$(5).jld2", "r") do f
#     JLD2.read(f, "results_dict")
# end
# p = jldopen("./data/results_dict_test-dur2000-numg$(5).jld2", "r") do f
#     JLD2.read(f, "params")
# end



# results = plot_results(results_dict, p)
# sort results from smallest to largest



# results = plot_results(results_dict, p)
# k1 = kde!(results)
# # plot with gadfly
# using KernelDensityEstimatePlotting
# show the plot


        # ax[ix].hist(value, bins=20, color=pcols[ix])
        # if ix==1
        #     ax[ix].axvline(starting_fitness, color="k", linestyle="dashed", linewidth=2, label="Starting Fitness")
        #     ax[ix].axvline(mean(value), color="r", linestyle="dashed", linewidth=2, label="Mean")
        # else
        #     ax[ix].axvline(starting_fitness, color="k", linestyle="dashed", linewidth=2)
        #     ax[ix].axvline(mean(value), color="r", linestyle="dashed", linewidth=2)
        # end
        # set xlims
#         max_val = maximum(arr)
#         # round up to  nearest 0.05
#         max_val = ceil(max_val * 20) / 20
#         # max_val = round(max_val, digits=2)
#         ax[ix].set_xlim(x_0 , max_val)
#         # xlims = ax[ix].get_xlim()
#         # if xlims[1] < 0.0
#         #     ax[ix].axes[:set_xticks]([0, round(xlims[2] ;digits=3)])
#         # else
#         #     ax[ix].axes[:set_xticks]([round(xlims[1] ; digits=3), round(xlims[2] ;digits=3)])
#         # end
#         # ax[ix].axes[:set_xticks](max_val)
#         ax[ix].axes[:set_yticks]([])
#         # set y axis title
#         ax[ix].set_ylabel("($(ix))  ")
#         # rotate y axis title
#         ax[ix].yaxis.label.set_rotation(0)
#     end
#     fig.legend(loc="upper center")
#     fig.suptitle("Fitness Distribution with Duration $(p[:duration])")
#     savefig("./images/oscillation_random-$(p[:num_random])genomes-home")
# # end # end main, uncomment to run as a script. keep commented to run in REPL
# # save results_dict as a .jld2 file
# JLD2.@save "results_dict_duration1000.jld2" results_dict
# # get fitness of the first key in results_dict
# find index of 0.1 in p[:fitness_chunks]
