using JLD2
using PyPlot
using Statistics

files = readdir("./data/batch3/")
file = jldopen("./data/batch3/$(files[1])")
learnedFitnessAll = []
evolvedFitnessAll = []
for i in 1:length(files)
    file = jldopen("./data/batch3/$(files[i])")
    learnedFitness = file["learned/track"]
    evolvedFitness = file["evolved/track"]
    params = file["params"]
    subplot(1, 2, 1)
    if i ==1
        for j in 1:size(learnedFitness, 1)-1
            plot(learnedFitness[j, :], color="blue", alpha=0.4)
            plot(evolvedFitness[j, :], color="red", alpha=0.4)
        end
        plot(learnedFitness[end, :], color="blue", alpha=0.4, label="Evolution with learning")
        plot(evolvedFitness[end, :], color="red", alpha=0.4, label="Only evolution")
    else
        for j in 1:size(learnedFitness, 1)
            plot(learnedFitness[j, :], color="blue", alpha=0.4)
            plot(evolvedFitness[j, :], color="red", alpha=0.4)
        end
    end
# combine all files into learnedFitness and evolvedFitness
    push!(learnedFitnessAll, learnedFitness)
    push!(evolvedFitnessAll, evolvedFitness)
end
learnedFitnessAll = vcat(learnedFitnessAll...)
evolvedFitnessAll = vcat(evolvedFitnessAll...)
plt.legend()
subplot(1, 2, 2)
plot(transpose(mean(learnedFitnessAll, dims=1)), color="blue", label="Mean evolution with learning")
plot(transpose(mean(evolvedFitnessAll, dims=1)), color="red", label="Mean just evolution")
plt.legend()
