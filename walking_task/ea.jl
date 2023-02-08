include("./ctrnn.jl")
include("./leggedwalker.jl")
include("./fitness_function.jl")
using Random
using Distributions
#define a microbial algorithm to evolve a legged walker
#using the genetic algorithm

struct MicrobParams
    #parameters
    popsize::Int64 #number of microbes
    genesize::Int64 #number of genes
    N::Int64 #number of neurons
    demesize::Int64 #number of microbes in a deme
    mutateProb::Float64 #mutation probability
    recombProb::Float64
    generations::Int64 #number of generations
    generator::String
    configuration::Int64
end

struct MicrobParamsIsland
    #parameters
    popsize::Int64 #number of microbes
    genesize::Int64 #number of genes
    N::Int64 #number of neurons
    islandN::Int64 #number of islands
    mutateProb::Float64 #mutation probability
    recombProb::Float64
    generations::Int64 #number of generations
    generator::String
    configuration::Int64
end
mutable struct MicrobePop   #changing parameters within population
    pop::Array{Float64,2} #microbes
    f::Array{Float64,1} #fitness
    best::Float64 #best fitness
    bestIndex::Int64 #index of best fitness
    bestx::Array{Float64,1} #best solution
    bestTrack::Array{Float64,1} #best fitness over time
    avgTrack::Array{Float64,1} #average fitness over time
    currentGen::Int64
end

MicrobeParams(popsize, genesize, N, demesize, mutateProb,
              recombProb, generations,
              generator, configuration) =
    MicrobParams(popsize, genesize, N, demesize,
                 mutateProb, recombProb, generations, generator,
                 configuration)

MicrobeParamsIsland(popsize, genesize, N, islandN, mutateProb,
              recombProb, generations,
              generator, configuration) =
    MicrobParamsIsland(popsize, genesize, N, islandN,
                 mutateProb, recombProb, generations, generator,
                 configuration)
MicrobePop(popsize, genesize, generations) = MicrobePop(rand(popsize, genesize).* 2 .- 1,
                                              zeros(popsize), 0.0,0, zeros(genesize),
                                              zeros(generations), zeros(generations), 1)

function createMicrobial(popsize::Int64,N::Int64, demesize::Int64,
                         recombProb::Float64, mutateProb::Float64,
                         generations::Int64, generator::String, configuration::Int64)
    genesize = N*N+2*N
    params = MicrobParams(popsize, genesize, N, demesize-1, mutateProb,
                           recombProb, generations, generator, configuration)
    pop = MicrobePop(popsize, genesize, generations)
    return (params,pop)
end

function createMicrobialIsland(popsize::Int64,N::Int64, islandN::Int64,
                         recombProb::Float64, mutateProb::Float64,
                         generations::Int64, generator::String, configuration::Int64)
    genesize = N*N+2*N
    params = MicrobParamsIsland(popsize, genesize, N, islandN, mutateProb,
                           recombProb, generations, generator, configuration)
    pop = MicrobePop(popsize, genesize, generations)
    return (params,pop)
end
function fitstats(microbial)
    params, pop = microbial
    bestFit, bestIndex = findmax(pop.f)
    if isnan(bestFit)
        throw("Fitness is Nan")
    end
    pop.bestx = pop.pop[bestIndex,:]
    avgFit = mean(pop.f)
    pop.bestTrack[pop.currentGen] = bestFit
    pop.avgTrack[pop.currentGen] = avgFit
    println("Generation $(pop.currentGen)")
    println("------------------------")
    println("Best idx: $(bestIndex) with fitness $(bestFit)")
    println("Average fitness: $(avgFit)")
end

function mutate(microbial, loser)
    params = microbial[1]
    pop = microbial[2]
    for i in 1:params.genesize
        # muate each gene with normal distribution centered at 0 and sd mutateProb
        pop.pop[loser,i] += rand(Normal(0,1))*params.mutateProb
    end
    clamp!(pop.pop[loser,:], -1, 1)
    pop.f[loser] = fitness_function(pop.pop[loser,:], params.N, params.generator, params.configuration)
end

function recombine(microbial, loser, winner)
    params = microbial[1]
    pop = microbial[2]
    for i in 1:params.genesize
        if rand() < params.recombProb
            pop.pop[loser,i] = pop.pop[winner,i]
        end
    end
end

function recombine(params, pop, loser, winner)
    for i in 1:params.genesize
        if rand() < params.recombProb
            pop.pop[loser,i] = pop.pop[winner,i]
        end
    end
end

function assignFitness(microbial, index)
    params = microbial[1]
    pop = microbial[2]
    pop.f[index] = fitness_function(pop.pop[index,:], params.N, params.generator, params.configuration)
end

function runMicrobial(microbial)
    params = microbial[1]
    pop = microbial[2]
    #assign fitness
    for i in 1:params.popsize
        assignFitness(microbial, i)
    end
    for i in 1:params.generations
        println("Generation: ", i,)
        maxFit, maxIndex = findmax(pop.f)
        println("Best fitness: ", maxFit)
        fitstats(microbial)
        for j in 1:params.popsize
            # pick two random individuals
            m1 = rand(1:params.popsize)
            m2 = mod1(rand(m1 - params.demesize:m1+params.demesize), params.popsize)
            while m1 == m2
                m2 = mod1(rand(m1 - params.demesize:m1+params.demesize), params.popsize)
            end
            if pop.f[m1] > pop.f[m2]
                winner = m1
                loser = m2
            else
                winner = m2
                loser = m1
            end
            recombine(microbial, loser, winner)
            mutate(microbial, loser)
        end
        pop.currentGen += 1
    end
end


function createIslands(popsize::Int64, islandN::Int64)
    indices = collect(1:popsize)
    islands = Array{Array{Int64,1},1}(undef, islandN)
    for i in 1:islandN
        islands[i] = indices[1:popsize÷islandN]
        indices = indices[popsize÷islandN+1:end]
    end
    return islands
end
function runMicrobialIsland(microbial)
    params = microbial[1]
    pop = microbial[2]
    #assign fitness
    for i in 1:params.popsize
        assignFitness(microbial, i)
    end
    for i in 1:params.generations
        println("Generation: ", i,)
        maxFit, maxIndex = findmax(pop.f)
        println("Best fitness: ", maxFit)
        fitstats(microbial)
        islands = createIslands(params.popsize, params.islandN)
        for island in islands
            shuffle!(island)
            pairs = reshape(island, 2, :)
            for j in 1:size(pairs)[1]
                if pop.f[pairs[j,1]] > pop.f[pairs[j,2]]
                    winner = pairs[j,1]
                    loser = pairs[j,2]
                else
                    winner = pairs[j,2]
                    loser = pairs[j,1]
                end
                recombine(params, pop, loser, winner)
                mutate(microbial, loser)
            end
        end
        for i in 1:params.popsize
            assignFitness(microbial, i)
        end
        pop.currentGen += 1
    end
end
microbial = createMicrobialIsland(100, 3, 10, 0.5, 0.05, 100, "CPG", 1)
runMicrobialIsland(microbial)
params, pop = microbial
# @code_warntype runMicrobial(microbial)
