using LinearAlgebra
using Distributions
include("./ctrnn.jl")



# For a heterogenous system it is possible to turn each parameter into a value per fluctuating variable
# This specific implementation is not heterogenous and will only fluctuate
# See python version of the class LRule in https://github.com/ajclaros/rl_legged_walker/matrix_rl/learning_rule.py


struct LRuleParams
    learn_rate
    conv_rate
    performance_bias
    tolerance
    param_min
    param_max
    period_min::Number
    period_max::Number
    init_flux
    max_flux
    flux_sign

end
mutable struct LRule
    center_mat::Array{Float64,2}
    extended_mat::Array{Float64,2}
    displacement_mat::Array{Float64,2}
    flux_mat::Array{Float64,2}
    moment_mat::Array{Float64,2}
    period_mat::Array{Float64,2}
    t
end

# Uniform learn_rate, conv_rate, parameter_min, parameter_max, init_flux, max_flux, and period ranges
function createLRuleHomogenous(learn_rate::Float64, conv_rate::Float64,
                                 performance_bias::Float64, tolerance::Float64,
                                 param_min::Float64, param_max::Float64,
                                 period_min::Int64, period_max::Int64,
                                 init_flux::Float64, max_flux::Float64,
                                 center_mat::Array{Float64,2}
                                 )
    params = LRuleParams(learn_rate, conv_rate, performance_bias, tolerance,
                param_min, param_max, period_min, period_max, init_flux, max_flux, create_random_matrix(size(center_mat)))

    flux_mat =  init_flux .* ones(size(center_mat))
    moment_mat = zeros(size(center_mat))
    period_mat = zeros(size(center_mat))
    learner = LRule(center_mat, center_mat, zeros(size(center_mat)),flux_mat, moment_mat, period_mat, 0)
    return (params, learner)
end


"a: scale, b:shift, c:growth rate"
function gompertz(x,; a=0.1, b=7, c=0.1)
    return a*exp(-b*exp(-c*x))
end

function iter_moment(dt::Float64, params::LRuleParams, learner::LRule)
    learner.moment_mat .+= dt
    indices = findall(learner.moment_mat .> learner.period_mat)
    learner.moment_mat[indices] .= 0
    #learner.period_mat[indices] .= rand.(Uniform(params.period_min, params.period_max))
    flux_period_center = (params.period_max+params.period_min)/2
    dev = (params.period_max-params.period_min)/4
    learner.period_mat[indices] .= randn(length(indices)) .* dev .+ flux_period_center
end

function update_weights_with_reward(reward, params, matrices)
    if abs(reward) >= params.tolerance
        matrices.extended_mat = matrices.center_mat .+ matrices.displacement_mat
        clamp!(matrices.extended_mat, params.param_min, params.param_max)
        matrices.center_mat .+= (matrices.extended_mat .- matrices.center_mat) .* (params.learn_rate *reward)
        clamp!(matrices.center_mat, params.param_min, params.param_max)
        matrices.flux_mat .-= params.conv_rate * reward# .*params.flux_sign
        clamp!(matrices.flux_mat, 0, params.max_flux)
        matrices.displacement_mat .= matrices.flux_mat.* sin.((matrices.moment_mat ./ matrices.period_mat) .* (2*pi))
        if matrices.t > 1
            #matrices.flux_mat .-= gompertz(matrices.t, 0.55, 5, 0.01)
            # matrices.flux_mat .-= gompertz(matrices.t)
            # clamp!(matrices.flux_mat, 0, params.max_flux)
            matrices.t = 1
        end
    else
        #matrices.flux_mat .+= gompertz(matrices.t, 0.55, 5, 0.01)
        matrices.flux_mat .+= matrices.flux_mat.*gompertz(matrices.t)
        clamp!(matrices.flux_mat, 0, params.max_flux)
        matrices.t += 1
    end
end



createLRuleHomogenous() = createLRuleHomogenous(0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 5, 10, 0.1, 0.1, zeros(3,5))
createLRuleHomogenous(center_mat::Array{Float64,2}) = createLRuleHomogenous(0.1, 0.1,
                                                                            0.0, 0.0,
                                                                            -16.0, 16.0,
                                                                            5, 10,
                                                                            1.0, 12.0,
                                                                            center_mat)
function create_random_matrix(size::Tuple{Int64,Int64})
    return rand([-1, 1], size[1], size[2])
end

# x = createLRuleHomogenous()
# @code_warntype iter_moment( 0.01, x[1], x[2])
# @code_warntype update_weights_with_reward(0.1, x[1], x[2])
