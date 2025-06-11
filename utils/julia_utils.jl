using DataFrames

function noise_v_collocation_points(
                                sys,#::ODESystem, # ODE system
                                sol,#::Union{ODESolution, Dict}, # Solution object or dictionary of solutions
                                noise_vals::Vector, # Noise levels to test
                                n_data_points::Vector; # Number of data points to select from the solution
                                n_iter::Int = 20, # Number of iterations for averaging the estimates
                                )
    u = sol.u
    t = sol.t

    return noise_v_collocation_points(sys, u, t, noise_vals, n_data_points; n_iter=n_iter)
end


"""
    noise_v_collocation_points(sys, sol, noise_vals, n_data_points, n_iter)
Estimate parameters of an ODE system using Physics-Informed Regression with varying noise levels and data points.
This function takes an ODE system, a solution object or dictionary of solutions, a vector of noise levels, a vector of number of data points, and the number of iterations for averaging the estimates.

# Args: \n
- `sys`::ODESystem: The ODE system to analyze.
- `sol`::Union{ODESolution, Dict}: The solution object or a dictionary of solutions.
- `noise_vals`::Vector{Float64}: A vector of noise levels to test.
- `n_data_points`::Vector{Int}: A vector of the number of data points to select from the solution.
- `n_iter`::Int: The number of iterations for averaging the estimates (default is 20).
# Returns: \n
- `Dict{Tuple, Vector}`: A dictionary containing the parameters for each combination of noise level and number of data points.
"""
function noise_v_collocation_points(
                                sys,#::ODESystem, # ODE system
                                u::Array,#::Union{ODESolution, Dict}, # Solution object or dictionary of solutions
                                t::Array,#::Vector{Float64}, # Time vector
                                noise_vals::Vector,# Noise levels to test
                                n_data_points::Vector; # Number of data points to select from the solution
                                n_iter::Int = 20,# Number of iterations for averaging the estimates
                                )
    max_u_val = maximum(abs.(hcat(u...)), dims=2)
    total_n_data_points = length(t)
    u0 = u[1,:] # Initial condition
    parameter_estimates = Dict{Tuple{Int,Float64}, Vector{Float64}}()
    for noise in noise_vals
        for n_data_points in n_data_points
            # Select a subset of the solution

            param_ests = zeros(length(parameters(sys)))
            for _ in 1:n_iter
                grid_step_size = ceil(Int, total_n_data_points / n_data_points)

                selected_t = t[1:grid_step_size:end]
                selected_u = (hcat(u[1:grid_step_size:end, :]...)+randn(length(u0),length(selected_t)).* max_u_val .* noise)'

                #reshape
                selected_u = [selected_u[i,:] for i in 1:size(selected_u,1)]

                # Compute the derivatives using spline interpolation
                du_approx = PhysicsInformedRegression.finite_diff(selected_u, selected_t)

                # Estimate the parameters
                paramsest = PhysicsInformedRegression.physics_informed_regression(sys, selected_u, du_approx)
                param_ests .+= [paramsest[param] for param in parameters(sys)]
            end
            param_ests ./= n_iter # Average the estimates over the 20 iterations

            #compute relative error for each parameter
            #relative_errors = [abs.((param_ests[i] - p[parameters(sys)[i]]) / p[parameters(sys)[i]]) for i in 1:length(parameters(sys))]

            # Store the estimates
            parameter_estimates[(n_data_points,noise)] = param_ests
        end
    end
    return parameter_estimates
end



"""
    create_table(rel_errors::Dict{Tuple, Vector}) -> DataFrame

Create a DataFrame from the parameter estimates dictionary, organizing the relative errors of the estimated parameters by noise level and number of data points.
# Args:
- `rel_errors`::Dict{Tuple, Vector}: A dictionary where keys are tuples of (number of data points, noise level) and values are vectors of relative errors for the estimated parameters.
# Returns:
- `DataFrame`: A DataFrame with rows corresponding to the number of data points and columns corresponding to different noise levels, containing the relative errors of the estimated parameters.
"""
function create_table(rel_errors; parameter_idx = 1)
    # Collect unique values for rows and columns
    n_data_points_vals = sort(unique([k[1] for k in keys(rel_errors)]))
    noise_vals = sort(unique([k[2] for k in keys(rel_errors)]))

    # Build a matrix of relative errors for α
    err_matrix = round.([rel_errors[(n, noise)][parameter_idx ] for n in n_data_points_vals, noise in noise_vals]*100, sigdigits=4)

    # Create DataFrame
    df = DataFrame(err_matrix, :auto)
    rename!(df, Symbol.(string.("noise_", noise_vals)))
    df.n_data_points = n_data_points_vals
    select!(df, :n_data_points, Not(:n_data_points))  # Move n_data_points to first column
    return df
end


