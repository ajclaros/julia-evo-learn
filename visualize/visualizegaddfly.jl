using Base: _push_deleted!
using JLD2
using Statistics
using Gadfly
import Cairo
import Fontconfig
folder = "../simple_oscillator_task/data/batch3/"
files = readdir(folder)
learnedFitness = []
evolvedFitness = []
afterLearnedFitness = []
# concat all the data
for i in 1:length(files)
    data = jldopen(folder * files[i])
    push!(learnedFitness, data["learned/track"])
    push!(evolvedFitness, data["evolved/track"])
    push!(afterLearnedFitness, data["learned/trackAfterLearn"])
end
# create Rdataset of time series
learnedFitness = vcat(learnedFitness...)
evolvedFitness = vcat(evolvedFitness...)
afterLearnedFitness = vcat(afterLearnedFitness...)
# plot individual time series by rows
learnedPlot = [layer(x=1:size(learnedFitness, 2), y=learnedFitness[i, :], Geom.line, Theme(default_color=colorant"blue")) for i in 1:size(learnedFitness, 1)]
evolvedPlot = [layer(x=1:size(evolvedFitness, 2), y=evolvedFitness[i, :], Geom.line, Theme(default_color=colorant"red")) for i in 1:size(evolvedFitness, 1)]
# add legend
individualRuns = plot(learnedPlot..., evolvedPlot..., Guide.xlabel("Generation"), Guide.ylabel("Fitness"), Guide.title("Fitness over time"), Guide.manual_color_key("Legend", ["Learned", "Evolved", "Learned after Evolved"], [colorant"blue", colorant"red", colorant"green"]), Coord.cartesian(xmin=0, xmax=300, ymin=0.0, ymax=0.85))


# mean and std of time series
meanLearnedFitness = mean(learnedFitness, dims=1)
stdLearnedFitness = std(learnedFitness, dims=1)/sqrt(size(learnedFitness, 1))
meanPlusSTD = meanLearnedFitness .+ stdLearnedFitness
meanMinusSTD = meanLearnedFitness .- stdLearnedFitness


meanEvolvedFitness = mean(evolvedFitness, dims=1)
stdEvolvedFitness = std(evolvedFitness, dims=1)/sqrt(size(evolvedFitness, 1))
meanPlusSTDEvolved = meanEvolvedFitness .+ stdEvolvedFitness
meanMinusSTDEvolved = meanEvolvedFitness .- stdEvolvedFitness

meanAfterLearnedFitness = mean(afterLearnedFitness, dims=1)
stdAfterLearnedFitness = std(afterLearnedFitness, dims=1)/sqrt(size(afterLearnedFitness, 1))
meanPlusSTDAfterLearned = meanAfterLearnedFitness .+ stdAfterLearnedFitness

meanPlot = plot(
 Coord.cartesian(xmin=0, xmax=300, ymin=0.0, ymax=0.85),
    Guide.xlabel("Generation"), Guide.ylabel("Fitness"),
    Guide.title("Mean Fitness over time"),
    Guide.manual_color_key("Legend", ["Learned", "Evolved"],
                           [colorant"blue", colorant"red"]))

push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanLearnedFitness,
                      Geom.line, Theme(default_color=colorant"blue")))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanPlusSTD, Geom.line, Theme(default_color=colorant"gray", line_style=[:ldash])))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanMinusSTD, Geom.line, Theme(default_color=colorant"gray", line_style=[:dash])))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), ymax=meanPlusSTD, ymin=meanMinusSTD, Geom.ribbon, Theme(default_color=colorant"blue")),)

push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanEvolvedFitness, Geom.line, Theme(default_color=colorant"red")))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanPlusSTDEvolved, Geom.line, Theme(default_color=colorant"gray", line_style=[:ldash])))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanMinusSTDEvolved, Geom.line, Theme(default_color=colorant"gray", line_style=[:dash])))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), ymax=meanPlusSTDEvolved, ymin=meanMinusSTDEvolved, Geom.ribbon, Theme(default_color=colorant"red")))

hstack(individualRuns, meanPlot) |>PDF("batch4.pdf", 30cm, 10cm)
