using Distributed
using NPZ
using JLD2
@everywhere begin
    include("./ctrnn.jl")
    include("./leggedwalker.jl")
    include("./rl_ctrnn.jl")
    include("./fitness_function.jl")
    include("./learning_function.jl")
    include("./ea.jl")

function runLearningTrials(idx, genome,
                              ;params_duration=5000,
                               params_size=2,
                               params_generator="CPG",
                               params_config=3,
                               params_num_trials=5,
                               params_window_size=220,
                               params_learn_rate=0.1,
                               params_conv_rate=0.000000001,
                               params_init_flux=0.02,
                               params_max_flux=0.2,
                               params_period_min=440,
                               params_period_max=4400,
                               params_learning_start=3000,
                               params_dt=0.1,
                              )
    end_fitnesses = zeros(params_num_trials)
    Threads.@threads for i in 1:params_num_trials
        end_fitnesses[i] = learn(genome,
                                 params_duration=params_duration,
                                 params_size= params_size,
                                 params_generator=params_generator,
                                 params_config=params_config,
                                 params_window_size=params_window_size,
                                 params_learn_rate=params_learn_rate,params_conv_rate= params_conv_rate,
                                 params_init_flux=params_init_flux,params_max_flux=params_max_flux,
                                 params_period_min=params_period_min, params_period_max=params_period_max,
                                 params_learning_start=params_learning_start, params_dt=params_dt)
    end
    return mean(end_fitnesses)
end

mutable struct MicrobePopLearn   #changing parameters within population
    pop::Array{Float64,2} #microbes
    f::Array{Float64,1} #fitness
    best::Float64 #best fitness
    bestIndex::Int64 #index of best fitness
    bestx::Array{Float64,1} #best solution
    bestTrack::Array{Float64,1} #best fitness over time
    avgTrack::Array{Float64,1} #average fitness over time
    currentGen::Int64
    learnedFitness::Array{Float64,1}
    bestTrackLearned::Array{Float64,1}
    avgTrackLearned::Array{Float64,1}
end

MicrobPopLearn(popsize, genesize, generations) =
    MicrobePopLearn(rand(popsize, genesize).*2 .-1,
                    zeros(popsize), 0.0, 0, zeros(genesize),
                    zeros(generations), zeros(generations), 1,
                    zeros(popsize), zeros(generations), zeros(generations))

function createMicrobialLearn(popsize::Int64, N::Int64, demesize::Int64,
                              recombProb::Float64,
                              mutateProb::Float64,
                              generations::Int64,
                              generator::String,
                              config::Int64,
                              )
    genesize = N*N + 2*N + 1
    params = MicrobParams(popsize,
                          genesize, N,
                          demesize-1,
                          mutateProb,
                          recombProb,
                          generations,
                          generator, config)
    pop = MicrobPopLearn(popsize, genesize, generations)
    return (params, pop)
end

function mutateLearn(microbial, loser)
    params = microbial[1]
    pop = microbial[2]
    for i in 1:params.genesize
        # muate each gene with normal distribution centered at 0 and sd mutateProb
        pop.pop[loser,i] += rand(Normal(0,1))*params.mutateProb
    end
    pop.pop[loser,:] = clamp.(pop.pop[loser,:], -1, 1)
    pop.f[loser] = fitness_function(pop.pop[loser,:], params.N, params.generator, params.configuration)
end

function fitstatsLearn(pop::MicrobePopLearn)
    bestFit, bestIndex = findmax(pop.f)
    bestLearned, bestLearnedIndex = findmax(pop.learnedFitness)
    averageFit = mean(pop.f)
    averageLearned = mean(pop.learnedFitness)
    pop.bestTrack[pop.currentGen] = bestFit
    pop.avgTrack[pop.currentGen] = averageFit
    pop.bestTrackLearned[pop.currentGen] = bestLearned
    pop.avgTrackLearned[pop.currentGen] = averageLearned
    # for i in 1:length(pop.f)
    #     fitness = pop.f[i]
    #     learned = pop.learnedFitness[i]
    #     println("Fit: $fitness, learned: $learned")
    # end
    println("Generation $(pop.currentGen)")
    println("------------------------")
    println("Best idx: $(bestIndex) with fitness $(bestFit)")
    println("Best learned idx: $(bestLearnedIndex) with fitness $(bestLearned)")
    println("Average fitness: $(averageFit)")
    println("Average learned fitness: $(averageLearned)")
end


function setFitness(params,pop, index)
    params = microbial[1]
    pop = microbial[2]
    pop.f[index] = fitness_function(pop.pop[index,:], params.N, params.generator, params.configuration)
end
function getFitnessIndex(pop::MicrobePopLearn, index::Int64, params::MicrobParams)
    pop.f[index] = fitness_function(pop.pop[index,:], params.N, params.generator, params.configuration)
end
end
function runMicrobialWithLearn(microbial
                              ;params_duration=5000,
                               params_size=3,
                               params_generator="CPG",
                               params_config=3,
                               params_num_trials=1,
                               params_window_size=220,
                               params_learn_rate=0.1,
                               params_conv_rate=0.00000001,
                               params_init_flux=0.02,
                               params_max_flux=0.2,
                               params_period_min=440,
                               params_period_max=4400,
                               params_learning_start=550,
                               params_dt=0.1
                              )
    params = microbial[1]
    pop = microbial[2]
    #assign fitnessos
    results = pmap(i -> runLearningTrials(i, pop.pop[i,:],
                                          params_duration=params_duration,
                                          params_size=params_size,
                                          params_generator=params_generator,
                                          params_config=params_config,
                                          params_num_trials=params_num_trials,
                                          params_window_size=params_window_size,
                                          params_learn_rate=params_learn_rate,
                                          params_conv_rate=params_conv_rate,
                                          params_init_flux=params_init_flux,
                                          params_max_flux=params_max_flux,
                                          params_period_min=params_period_min,
                                          params_period_max=params_period_max,
                                          params_learning_start=params_learning_start,
                                          params_dt=params_dt
                                         ), 1:params.popsize)
    for i in 1:params.popsize
        pop.learnedFitness[i] = results[i]
        getFitnessIndex(pop, i, params)
    end
    for i in 1:params.generations
        fitstatsLearn(pop)
        # prevent race conditions when overwriting genomes
        for j in 1:params.popsize
            # pick two random individuals
            m1 = rand(1:params.popsize)
            m2 = mod1(rand(m1-params.demesize:m1+params.demesize), params.popsize)
            while m1 == m2
                m2 = mod1(rand(m1 - params.demesize:m1+params.demesize), params.popsize)
            end
            if pop.learnedFitness[m1] > pop.learnedFitness[m2]
                winner = m1
                loser = m2
            else
                winner = m2
                loser = m1
            end
            recombine(params, pop, loser, winner)
            mutateLearn(microbial, loser)
        end
        results = pmap(j -> runLearningTrials(j, pop.pop[j,:],
                                              params_duration=params_duration,
                                              params_size=params_size,
                                              params_generator=params_generator,
                                              params_config=params_config,
                                              params_num_trials=params_num_trials,
                                              params_window_size=params_window_size,
                                              params_learn_rate=params_learn_rate,
                                              params_conv_rate=params_conv_rate,
                                              params_init_flux=params_init_flux,
                                              params_max_flux=params_max_flux,
                                              params_period_min=params_period_min,
                                              params_period_max=params_period_max,
                                              params_learning_start=params_learning_start,
                                              params_dt=params_dt), 1:params.popsize)

        for j in 1:params.popsize
            pop.learnedFitness[j] = results[j]
        end
        pop.currentGen += 1
    end
end

function main()
    popsize = 100
    demesize = 2
    generations = 500
    rec_rate = 0.5
    mut_rate = 0.05
    num_trials = 1
    learning_params = Dict(
        :params_duration=>2000,
        :params_num_trials=>1,
        :params_size=>3,
        :params_generator=>"CPG",
        :params_configuration=>3,
        :params_window_size=>220,
        :params_learn_rate=>0.1,
        :params_conv_rate=>0.00000001,
        :params_init_flux=>0.02,
        :params_max_flux=>0.2,
        :params_period_min=>440,
        :params_period_max=>4400,
        :params_learning_start=>550,
        :params_dt=>0.1)
    learnedTrack_Fitness = zeros(num_trials, generations)
    learnedTrack_AvgFitness = zeros(num_trials, generations)
    learnedTrack_AfterLearn = zeros(num_trials, generations)
    learnedTrack_AvgAfterLearn = zeros(num_trials, generations)

    evolvedTrack_Fitness = zeros(num_trials, generations)
    evolvedTrack_AvgFitness = zeros(num_trials, generations)
    evolvedTrack_AfterLearn = zeros(num_trials, generations)
    evolvedTrack_AvgAfterLearn = zeros(num_trials, generations)
    println("Running with $(nworkers()) workers")
    println("Max threads: $(Threads.nthreads())")
    for i in 1:num_trials
        println("Run $i")
        microbial = createMicrobialLearn(popsize, learning_params[:params_size],demesize, rec_rate, mut_rate, generations, learning_params[:params_generator],
                                         learning_params[:params_configuration])

        runMicrobialWithLearn(microbial,
                                params_duration=learning_params[:params_duration],
                                params_num_trials=learning_params[:params_num_trials],
                                params_size=learning_params[:params_size],
                                params_generator=learning_params[:params_generator],
                                params_config=learning_params[:params_configuration],
                                params_window_size=learning_params[:params_window_size],
                                params_learn_rate=learning_params[:params_learn_rate],
                                params_conv_rate=learning_params[:params_conv_rate],
                                params_init_flux=learning_params[:params_init_flux],
                                params_max_flux=learning_params[:params_max_flux],
                                params_period_min=learning_params[:params_period_min],
                                params_period_max=learning_params[:params_period_max],
                                params_learning_start=learning_params[:params_learning_start],
                                params_dt=learning_params[:params_dt])
        learnedTrack_Fitness[i,:] = microbial[2].bestTrack
        learnedTrack_AvgFitness[i,:] = microbial[2].avgTrack
        learnedTrack_AfterLearn[i,:] = microbial[2].bestTrackLearned
        learnedTrack_AvgAfterLearn[i,:] = microbial[2].avgTrackLearned
        microb = createMicrobial(popsize, learning_params[:params_size], N, rec_rate, mut_rate, generations,
                                        learning_params[:params_generator],
                                        learning_params[:params_configuration])


        runMicrobial(microb)
        evolvedTrack_Fitness[i,:] = microb[2].bestTrack
        evolvedTrack_AvgFitness[i,:] = microb[2].avgTrack
        evolvedTrack_AfterLearn[i,:] = microb[2].bestTrackLearned
        evolvedTrack_AvgAfterLearn[i,:] = microb[2].avgTrackLearned
        # evolvedFitness[i,:] = microb[2].bestTrack
        # evolvedAvgFitness[i,:] = microb[2].avgTrack
    end
    rmprocs(workers())
    filenum=0
    # filename = "evolveVLearn$(filenum).jld2"
    filestring = "./data/microbial/batch1/evoLrn$(learning_params[:params_duration])-T"
    # filestring = "./data/batch3/test"
    filename = "$(filestring)$(filenum).jld2"
    #check if file exists
    if isfile(filename)
        while isfile(filename)
            filenum += 1
            filename = "$(filestring)$(filenum).jld2"
        end
    end
    println("saving to $filename")
    jldopen(filename, "w") do file
        learned = JLD2.Group(file, "learned")
        learned["track"] = learnedTrack_Fitness
        learned["trackAvg"] = learnedTrack_AvgFitness
        learned["trackAfterLearn"] = learnedTrack_AfterLearn
        learned["trackAvgAfterLearn"] = learnedTrack_AvgAfterLearn

        evolved = JLD2.Group(file, "evolved")
        evolved["track"] = evolvedTrack_Fitness
        evolved["trackAvg"] = evolvedTrack_AvgFitness
        evolved["trackAfterLearn"] = evolvedTrack_AfterLearn
        evolved["trackAvgAfterLearn"] = evolvedTrack_AvgAfterLearn
        params = JLD2.Group(file, "params")
        params["num_trials"] = num_trials
        params["generations"] = generations
        params["popsize"] = popsize
        params["islandN"] = N
        params["recombRate"] = rec_rate
        params["mutateRate"] = mut_rate
        params["N"] = learning_params[:params_size]
        learn_params = JLD2.Group(file, "learn_params")
        learn_params["duration"] = learning_params[:params_duration]
        learn_params["num_trials"] = learning_params[:params_num_trials]
        learn_params["size"] = learning_params[:params_size]
        learn_params["window_size"] = learning_params[:params_window_size]
        learn_params["learn_rate"] = learning_params[:params_learn_rate]
        learn_params["conv_rate"] = learning_params[:params_conv_rate]
        learn_params["init_flux"] = learning_params[:params_init_flux]
        learn_params["max_flux"] = learning_params[:params_max_flux]
        learn_params["period_min"] = learning_params[:params_period_min]
        learn_params["period_max"] = learning_params[:params_period_max]
        learn_params["learning_start"] = learning_params[:params_learning_start]
        learn_params["dt"] = learning_params[:params_dt]
    end
end
main()
