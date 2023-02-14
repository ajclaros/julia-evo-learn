using JLD2
using Statistics
using Gadfly
folder = "../simple_oscillator_task/data/batch3/"
files = readdir(folder)

file = jldopen(folder * "$(files[1])")
learnedFitness = file["learned/track"]
learnedFitnessAll = []
evolvedFitnessAll = []
afterLearning = []

plotlearned = plot()
plotlearnedAfter = []
tplotevolved = []
# create empty plot
for i in 1:length(files)
    file = jldopen(folder* "$(files[i])")
    learnedFitness = file["learned/track"]
    afterlearnedFitness = file["learned/trackAfterLearn"]
    evolvedFitness = file["evolved/track"]
    params = file["params"]
    # subplot(1, 2, 1)
    if i ==1
        for j in 1:size(learnedFitness, 1)-1
            # plot(learnedFitness[j, :], color="blue", alpha=0.4)
            # plot(evolvedFitness[j, :], color="red", alpha=0.4)
            push!(plotlearned,
                  layer(y=learnedFitness[1],
                        x=1:size(learnedFitness, 2),
                        Theme(default_color=colorant"blue",
                                         line_width=0.5mm)))

            # push!(plotlearnedAfter, afterlearnedFitness[j, :])
            # push!(plotevolved, evolvedFitness[j, :])
        end
        # plot(learnedFitness[end, :], color="blue", alpha=0.4, label="Evolution with learning")
        # plot(evolvedFitness[end, :], color="red", alpha=0.4, label="Only evolution")
    # else
    #     for j in 1:size(learnedFitness, 1)
    #         plot(learnedFitness[j, :], color="blue", alpha=0.4)
    #         plot(evolvedFitness[j, :], color="red", alpha=0.4)
    #     end
    end
# combine all files into learnedFitness and evolvedFitness
    # push!(learnedFitnessAll, learnedFitness)
    # push!(evolvedFitnessAll, evolvedFitness)
    # push!(afterLearning, afterlearnedFitness)
end
# learnedFitnessAll = vcat(learnedFitnessAll...)
# evolvedFitnessAll = vcat(evolvedFitnessAll...)
# afterLearning = vcat(afterLearning...)
# plt.legend()
# subplot(1, 2, 2)
# num_trials = length(files) * size(file["learned/track"], 1)
# plot(transpose(mean(learnedFitnessAll, dims=1)), color="blue", label="Mean evolution with learning")
# # add standard error to the previous plot

# plot(transpose(mean(learnedFitnessAll, dims=1)) .+ 2*transpose(std(learnedFitnessAll, dims=1)/sqrt(num_trials)), color="blue", alpha=0.5, linestyle="dashed")
# plot(transpose(mean(learnedFitnessAll, dims=1)) .- 2*transpose(std(learnedFitnessAll, dims=1)/sqrt(num_trials)), color="blue", alpha=0.5, linestyle="dashed")
# plot(transpose(mean(evolvedFitnessAll, dims=1)), color="red", label="Mean just evolution")
# plot(transpose(mean(evolvedFitnessAll, dims=1)) .+ 2*transpose(std(learnedFitnessAll, dims=1)/sqrt(num_trials)), color="red", alpha=0.5, linestyle="dashed")
# plot(transpose(mean(evolvedFitnessAll, dims=1)) .- 2*transpose(std(learnedFitnessAll, dims=1)/sqrt(num_trials)), color="red", alpha=0.5, linestyle="dashed")
# plot(transpose(mean(afterLearning, dims=1)), color="green", label="Mean evolution with learning after learning")
# plt.legend()


# plotlearned = hcat(plotlearned)
# x = Gadfly.plot(x=1:size(plotlearned, 2), y=plotlearned, Geom.line, Guide.xlabel("Generation"), Guide.ylabel("Fitness"), Guide.title("Evolution with learning"))
# plotlearnedAfter = []
# plotevolved = []
# plot generation vs fitness for all trials

# x = Gadfly.plot(
#     layer(x=1:size(plotlearned[1], 1), y=plotlearned[1], Geom.line, Theme(default_color=colorant"blue")),
#     layer(x=1:size(plotlearnedAfter[1], 1), y=plotlearnedAfter[1], Geom.line, Theme(default_color=colorant"green")),
#     layer(x=1:size(plotevolved[1], 1), y=plotevolved[1], Geom.line, Theme(default_color=colorant"red")),
#     Guide.xlabel("Generation"), Guide.ylabel("Fitness"), Guide.title("Fitness over generations"),
#     Guide.manual_color_key("Legend", ["Evolution with learning", "Evolution with learning after learning", "Only evolution"], [colorant"blue", colorant"green", colorant"red"])
# )
display(plotlearned)
