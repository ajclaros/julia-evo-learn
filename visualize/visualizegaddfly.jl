using Base: _push_deleted!
using JLD2
using Statistics
using Gadfly
import Cairo
import Fontconfig
task = "simple_oscillator_task"
batch="batch5"
folder = "../$(task)/data/$(batch)/"
files = readdir(folder)
duration = 500
filename= "evLrnIsland$(duration)"
data = load(folder * filename * "-T0.jld2")
files = filter(x -> occursin("evLrnIsland$(duration)", x), files)
datatest = jldopen(folder* files[1])
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
ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
individualRuns = plot(learnedPlot..., evolvedPlot..., Guide.xlabel("Generation"), Guide.ylabel("Fitness"), Guide.title("Fitness over time"), Guide.manual_color_key("Legend", ["Learned", "Evolved", "Learned after Evolved"], [colorant"blue", colorant"red", colorant"green"]), Coord.cartesian(xmin=0, xmax=datatest["params"]["generations"], ymin=0.0, ymax=1.0),
                      Guide.yticks(ticks=ticks)
                      )


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
meanMinusSTDAfterLearned = meanAfterLearnedFitness .- stdAfterLearnedFitness

meanPlot = plot(
 Coord.cartesian(xmin=0, xmax=datatest["params"]["generations"], ymin=0.0, ymax=1.00),
    Guide.xlabel("Generation"), Guide.ylabel("Fitness"),
    Guide.title("Mean Fitness over time"),
    Guide.yticks(ticks=ticks),
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

push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanAfterLearnedFitness, Geom.line, Theme(default_color=colorant"green")))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanPlusSTDAfterLearned, Geom.line, Theme(default_color=colorant"gray", line_style=[:ldash])))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), y=meanMinusSTDAfterLearned, Geom.line, Theme(default_color=colorant"gray", line_style=[:dash])))
push!(meanPlot, layer(x=1:size(meanLearnedFitness, 2), ymax=meanPlusSTDAfterLearned, ymin=meanMinusSTDAfterLearned, Geom.ribbon, Theme(default_color=colorant"green")))

title(hstack(individualRuns, meanPlot), "Does learning improve evolution?") |>PDF("../$(task)/images/$(batch)/$(batch)D-$(duration).pdf", 30cm, 10cm)
command = `pdftoppm -png -r 300 ../$(task)/images/$(batch)/$(batch)D-$(duration).pdf ../$(task)/images/$(batch)/$(batch)D-$(duration)`
run(command)
println("done")
