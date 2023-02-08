# Julia Implementation of a single legged walker
struct agentParams
    LegLength::Float64
    MaxLegForce::Float64
    ForwardAngleLimit::Float64
    BackwardAngleLimit::Float64
    MaxVelocity::Float64
    MaxTorque::Float64
    MaxOmega::Float64
    MaxFootDistance::Float64
end

mutable struct agentState
    cx::Float64
    cy::Float64
    vx::Float64
    footstate::Int64
    angle::Float64
    omega::Float64
    forward_force::Float64
    backward_force::Float64
    jointX::Float64
    jointY::Float64
    footX::Float64
    footY::Float64
    leg_length::Float64
end

function agentParams()
    return agentParams(15.0, 0.05, pi/6.0, -pi/6.0, 6.0, 0.5, 1.0, 20.0)
end

function createLeggedAgent()
    params = agentParams()
    cx = 0.0
    cy = 0.0
    vx = 0.0
    footstate = 0
    angle = params.ForwardAngleLimit
    omega = 0.0
    forward_force = 0.0
    backward_force = 0.0
    jointX = cx
    jointY = cy + 12.5
    footX = jointX + params.LegLength * sin(angle)
    footY = jointY + params.LegLength * cos(angle)
    leg_length = params.LegLength
    state = agentState(cx, cy, vx, footstate, angle, omega, forward_force, backward_force, jointX, jointY, footX, footY, leg_length)

    return (params, state)
end

function getState(agent)
    params, state = agent
    return [state.angle, state.omega, state.footstate]
end

function getAngleFeedback(agent)
    params, state = agent
    return state.angle * 5.0 / params.ForwardAngleLimit
end

function stepNWalker(agent::Tuple{agentParams, agentState}, stepsize::Float64, configuration::Int64, outputs)
    params = agent[1]
    state = agent[2]
    force = 0.0
    if configuration == 1
        if outputs[1] > 0.5
            state.footstate = 1
            state.omega = 0.0
            state.forward_force = 2 * (outputs[1] - 0.5) * params.MaxLegForce
            state.backward_force = 0.0
        else
            state.footstate = 0.0
            state.forward_force = 0.0
            state.backward_force = 2 * (0.5 - outputs[1]) * params.MaxLegForce
        end
    elseif configuration ==2
        if outputs[1] > 0.5
            state.footstate = 1
            state.omega = 0
        else
            state.footstate = 0
        end
        state.forward_force = outputs[1] * params.MaxLegForce
        state.backward_force = outputs[2] * params.MaxLegForce
    elseif configuration ==3 && length(outputs)>2
        if outputs[1] > 0.5
            state.footstate = 1
            state.omega = 0
        else
            state.footstate = 0
        end
        state.forward_force = outputs[2] * params.MaxLegForce
        try
            state.backward_force = outputs[3] * params.MaxLegForce
        catch BoundsError
            throw("Configured for 3 outputs, but got $(size(outputs))")
        end
    else
        throw("Invalid controller value (must be 1, 2, or 3)")
    end
    # Compute force applied to body
    f = state.forward_force - state.backward_force
    if state.footstate == 1
        if (state.angle >= params.BackwardAngleLimit&& state.angle <= params.ForwardAngleLimit)||
                (state.angle < params.BackwardAngleLimit && f < 0)||
                (state.angle > params.ForwardAngleLimit && f > 0)
            force = f
        end
    end
    # Update position and velocity
    state.vx += force * stepsize
    if state.vx < -params.MaxVelocity
        state.vx = -params.MaxVelocity
    end
    if state.vx > params.MaxVelocity
        state.vx = params.MaxVelocity
    end
    state.cx += state.vx * stepsize

    # Update angle and angular velocity
    state.jointX += state.vx * stepsize
    if state.footstate == 1
        temp_angle = atan(state.footX - state.jointX,
                                   state.footY - state.jointY)
        state.omega = (temp_angle - state.angle) / stepsize
        state.angle = temp_angle
    else
        state.vx = 0.0
        state.omega += params.MaxTorque * stepsize * (state.backward_force - state.forward_force)
        if state.omega < -params.MaxOmega
            state.omega = -params.MaxOmega
        end
        if state.omega > params.MaxOmega
            state.omega = params.MaxOmega
        end
        state.angle += state.omega * stepsize
        if state.angle < params.BackwardAngleLimit
            state.angle = params.BackwardAngleLimit
            state.omega = 0.0
        end
        if state.angle > params.ForwardAngleLimit
            state.angle = params.ForwardAngleLimit
            state.omega = 0.0
        end
        state.footX = state.jointX + state.leg_length * sin(state.angle)
        state.footY = state.jointY + state.leg_length * cos(state.angle)
    end
    # If foot is too far forward or behind, reset velocity
    if abs(state.cx - state.footX) > params.MaxFootDistance
        state.vx = 0.0
    end
end

