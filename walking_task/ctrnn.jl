using LinearAlgebra

struct CTRNN
    n ::Int64
    time_constants ::Vector{Float64}
    inv_time_constants ::Vector{Float64}
    biases::Vector{Float64}
    weights::Matrix{Float64}
    WR::Float64       # weight range
    BR::Float64       # bias range
    TR::Float64      # time constant range
    TA::Float64       # time constant amplitude
end

mutable struct ctrnnState
    inputs::Vector{Float64}
    voltages::Vector{Float64}
    outputs::Vector{Float64}
end

function dotProductRowColumn(a, b)
    return [sum(a[i, :] .* b) for i in 1:size(a)[1]]
end
# @code_warntype dotProductRowColumn([1.0 2.0; 3.0 4.0], [1.0 2.0; 3.0 4.0])

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function value_to_range(value, mult, add)
    return value * mult + add
end

function range_to_value(value, mult, add)
    return (value - add) / mult
end
value_to_range(value, mult) = value_to_range(value, mult, 0)
range_to_value(value, mult) = range_to_value(value, mult, 0)

function ctrnnState(ctrnn::CTRNN, voltages::Array{Float64,1})
    return ctrnnState(zeros(ctrnn.n), voltages, sigmoid.(voltages .+ ctrnn.biases))
end

function ctrnnStep(dt::Float64, ctrnn::CTRNN, state::ctrnnState)
    netinput = state.inputs .+ dotProductRowColumn(transpose(ctrnn.weights) , state.outputs)
    state.voltages += dt .* (ctrnn.inv_time_constants .* (netinput .- state.voltages))
    state.outputs .= sigmoid.(state.voltages .+ ctrnn.biases)
end

function mapWeights(arr, n, WR)
    return reshape(value_to_range.(arr[1:n^2], WR), n, n)
end

function mapBias(arr, BR)
    return value_to_range.(arr, BR)
end

function mapTimeConstant(arr, TR, TA)
    return arr .* TR .+ TA
end

function recoverWeight(arr::Array{Float64,2}, n, ;WR=16)
    return reshape(range_to_value.(transpose(arr), WR), n^2)
end

function recoverBias(arr, ;BR=16)
    return range_to_value.(arr, BR)
end
function recoverTimeConstant(arr, ;TR=5.0, TA=6.0)
    return (arr .- TA) ./ TR
end

function createNodes(genome::Vector{Float64}, n=2::Int64, WR=16.0::Float64, BR=16.0::Float64, TR=5.0::Float64, TA=6.0::Float64)
    weights = transpose(mapWeights(genome[1:n^2], n, WR))
    biases = mapBias(genome[n^2+1:n^2+n], BR)
    time_constants = mapTimeConstant(genome[n^2+n+1:n^2+2*n], TR, TA)
    return CTRNN(n,
                 time_constants, 1 ./ time_constants, biases, weights,
                 WR, BR, TR, TA)
end
createNodes(genome::Vector{Float64}, n::Int64) = createNodes(genome, n, 16.0, 16.0, 5.0, 6.0)

function createNervousSystem(genome::Vector{Float64}, n::Int64, WR=16.0::Float64, BR=16.0::Float64, TR=5.0::Float64, TA=6.0::Float64)
    nodes = createNodes(genome, n, WR, BR, TR, TA)
    state = ctrnnState(nodes, zeros(n))
    return (nodes, state)
end
createNervousSystem(genome::Vector{Float64}, n::Int64) = createNervousSystem(genome, n, 16.0, 16.0, 5.0, 6.0)
# @code_warntype createNervousSystem(testgenome, 3)

function runCTRNN(genome, n, duration=220.0, WR=16.0, BR=16.0, TR=5.0, TA=6.0, dt=0.1)
    ns = createNervousSystem(genome, n, WR, BR, TR, TA)
    time_ns = range(0, duration-dt, step=dt)
    data = zeros(length(time_ns), n)
    for i in 1:length(time_ns)
        ctrnnStep(dt, ns)
        #data[i, :] = ns[:state].outputs
    end
    return data
end
runCTRNN(genome, n) = runCTRNN(genome, n, 220, 16.0, 16.0, 5.0, 6.0, 0.1)
runCTRNN(genome, n, duration) = runCTRNN(genome, n, duration, 16.0, 16.0, 5.0, 6.0, 0.1)

function plotCTRNN(genome, n, WR=16.0, BR=16.0, TR=5.0, TA=5.0, steps=1000)
    ns = createCTRNN(genome, n,  WR, BR, TR, TA)
    ns_state = ctrnnState(ns, [0.0, 0.0, 0.0])
    outputs = zeros(steps, n)
    for i = 1:steps
        ctrnnStep(0.1, ns, ns_state)
        outputs[i, :] = ns_state.outputs
    end
    return outputs
end


testgenome = [
0.04774359502446968,
-0.683711607305986,
0.45036338411104737,
0.9721092700062304,
0.7891519578423444,
-0.00960243655211588,
-0.9358149684117485,
-0.8152701212733787,
0.6207119728559448,
0.28996795347325205,
0.3639871362038097,
-0.6154338363438252,
0.4644851766806753,
-0.4605993067803686,
-0.4491022368326481,]

# ns = createNervousSystem(testgenome, 3)
# @btime runCTRNN(testgenome, 3)
# @code_warntype ctrnnStep(0.1, ns[1], ns[2])
