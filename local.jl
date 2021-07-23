module Local

using Clp, JuMP, Parameters, LinearAlgebra
using ScikitLearn
using Statistics
@sk_import tree: DecisionTreeClassifier
@sk_import neural_network: MLPClassifier
@sk_import metrics: accuracy_score
@sk_import base: clone
# export Problem

@with_kw mutable struct Problem
    OracleClass
    Objective
    X::Array{Float64,2}
    C::Array{Float64,2}
    W::Array{Float64,2}
    W_op::Array{Float64,2}
    n_clusters::Int64
    
    N::Vector{Vector{Int64}}
    EPS::Float64
    model
    cap_type
end;

function p_n(P)  return size(P.X)[1] end;
function p_dx(P) return size(P.X)[2] end;
function p_dw(P) return size(P.C)[2] end;

function Problem(OracleClass, Objective, X, C; cap_type = :regret)
    function init_weights()
        d = size(C)[2]
        opt_w = Array{Float64}(undef, 0, d)
        opt_w_op = Array{Float64}(undef, 0, d)

        model = OracleClass(zeros(d))

        for i=1:size(C)[1]
            c = C[i,:]
            @objective(model, Min, Objective(c, model[:w]))
            optimize!(model)

            opt_w = [opt_w; value.(model[:w])']

            # @objective(model, Max, Objective(c, model[:w]))
            # optimize!(model)
            # opt_w_op = [opt_w_op; value.(model[:w])']
        end;
        return opt_w, opt_w_op
    end;

    cc = zeros(size(C)[1], size(C)[2])
    for i=1:size(C)[1]
        cc[i,:] = C[i,:] / norm(C[i,:])
    end
    W, W_op = init_weights()
    N = Vector{Vector{Int64}}()
    model = 0
    return Problem(OracleClass, Objective, X, cc, W, W_op, size(X)[1], N, -1, model, cap_type)
end;

function l1_dist(x, y)
    return sum(abs.(x .- y))
end

function init_n(P, EPS)
    P.EPS = EPS
    N = Vector{Vector{Int64}}()
    n = size(P.W)[1]
    for tt=1:P.n_clusters
#         i = rand(1:n)
        i = tt
        cur = Vector{Int64}()
        dists = Vector{Float64}()
        for k=1:n

            # d = P.Objective(P.C[i,:], P.W[k,:]) - P.Objective(P.C[i,:], P.W[i,:])
            # push!(dists, d)
    
            if P.cap_type == :regret
                d = P.Objective(P.C[i,:], P.W[k,:]) - P.Objective(P.C[i,:], P.W[i,:])
            elseif P.cap_type == :cap
                d = l1_dist(P.W[k,:], P.W[i,:])
            end

            if d < EPS
                push!(cur, k)
            end;
        end;
        # q = quantile!(dists, P.EPS)
        # for k = 1:n
        #     if dists[k] <= q
        #         push!(cur, k)
        #     end
        # end
        push!(N, cur)
    end;
    P.N = N
end;

function init_models(P, model_class)
    n = size(P.W)[1]
    model = Vector{Any}()
    for i=1:P.n_clusters
        labels = zeros(n)
        for j in P.N[i,:][1]
            labels[j] = 1
        end;
        
        weights = Vector{Float64}()
        for j = 1:P.n_clusters 
            if j in P.N[i,:][1]
                labels[j] = 1
                push!(weights, 1 - (P.Objective(P.C[i,:], P.W[j,:]) - P.Objective(P.C[i,:], P.W[i,:])))
            else 
               push!(weights, P.Objective(P.C[i,:], P.W[j,:]) - P.Objective(P.C[i,:], P.W[i,:])) 
            end
        end;

#         println(weights)
        m = clone(model_class)
#         m.fit(P.X, labels, sample_weight = weights)
        m.fit(P.X, labels)

        push!(model, m)
    end;
    P.model = model
end;

function predict(P, x; return_penalty = false, eps = -1)
    if eps == -1
        eps = P.EPS
    end

    model = P.OracleClass(zeros(p_dw(P)))
    MOI.set(model, MOI.Silent(), true)

    n = p_n(P)
    m = p_dw(P)

    cc = Vector{Int64}()
    r = Vector{Float64}()

    @variable(model, q[1:P.n_clusters] >= 0)
    
    if P.cap_type == :cap
        @variable(model, ab[1:n, 1:m])
    end

#     for i = 1:m
#         set_binary(model[:w][i])
#     end

    cur = 0
    for i=1:P.n_clusters
        p = P.model[i].predict_proba([x])
        if size(p)[2] == 1
            push!(r, 0)
            continue 
        end
    
        push!(r, p[2])
        
        if P.cap_type == :cap
            for j=1:m
                @constraint(model, ab[i,j] >= model[:w][j] - P.W[i,j])
                @constraint(model, ab[i,j] >= P.W[i,j] - model[:w][j])
            end
            @constraint(model, sum(ab[i,j] for j=1:m) - eps <= q[i])
        elseif P.cap_type == :regret
            @constraint(model, P.Objective(P.C[i,:], model[:w]) - P.Objective(P.C[i,:], P.W[i,:]) - eps <= q[i])
        end

    end;

    # println(r)
    @objective(model, Min, sum(q .* r))

    optimize!(model)
    if termination_status(model) == MOI.INFEASIBLE
        println("INFEASIBLE")
        return "INFESIBLE", cc
    end

    if return_penalty
        return value.(model[:w]), r
    end
    return value.(model[:w])
end;

end  # module Local
