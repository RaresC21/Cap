module LocalHousing

using Clp, JuMP, Parameters, LinearAlgebra
using ScikitLearn
using Statistics

@sk_import tree: DecisionTreeClassifier
@sk_import neural_network: MLPClassifier
@sk_import metrics: accuracy_score
@sk_import base: clone
# export Problem


@with_kw mutable struct ProblemNewsvendor
    h::Int64
    b::Int64

    X::Array{Float64,2}
    D::Array{Float64,1}

    N::Vector{Vector{Int64}}
    EPS::Float64
    model
    epsilons 
end;

function ProblemNewsvendor(h, b, X, D)
    N = Vector{Vector{Int64}}()
    return ProblemNewsvendor(h, b, X, D, N, -1, 0, 0)
end;

function p_n(P)  return size(P.X)[1] end;
function p_dx(P) return size(P.X)[2] end;
function p_dw(P) return size(P.C)[2] end;

function init_n(P, EPS)
    P.EPS = EPS
    N = Vector{Vector{Int64}}()
    epsilons = Vector{Float64}()
    n = size(P.D)[1]
    for tt=1:n
#         i = rand(1:n)
        i = tt
        cur = Vector{Int64}()
        dists = Vector{Float64}()
        for k=1:n
            
#             d = max(P.h * (P.D[k] - P.D[i]), P.b * (P.D[i] - P.D[k]))
#             push!(dists, d)
            
            if P.D[k] >= P.D[i] - EPS / P.b && P.D[k] <= P.D[i] + EPS / P.h
                push!(cur, k)
            end;
        end;
        
#         q = quantile!(dists, P.EPS)
#         push!(epsilons, q)
#         for k = 1:n
#             if dists[k] <= q
#                 push!(cur, k)
#             end
#         end

        
        push!(N, cur)
    end;
    P.epsilons = epsilons
    P.N = N
end;

function init_models(P, model_class)
    n = size(P.D)[1]
    model = Vector{Any}()
    for i=1:n
#         if i % 1 == 0
#             println(i)
#         end
        labels = zeros(n)
        for j in P.N[i,:][1]
            labels[j] = 1
        end;

        m = clone(model_class)
    
        m.fit(P.X, labels)

        pred = m.predict(P.X)
#         println(i, " ", accuracy_score(pred, labels), " ", 1 - sum(labels) / n)

        push!(model, m)
    end;
    P.model = model
end;

function predict(P, x; return_penalty = false, eps = -1)
    if eps < -0.5
        eps = P.EPS
    else
        eps = 0
    end
    
    model = Model(Clp.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    n = p_n(P)
    cc = Vector{Int64}()
    r = Vector{Float64}()

    @variable(model, z)
    @variable(model, s[1:n])
    @variable(model, q[1:n] >= 0)
    @variable(model, high[1:n], Bin)
    
    for i=1:n
        p = P.model[i].predict_proba([x])
#         println(p)
        if size(p)[2] == 1
            push!(r, 1)
            continue
        end
        push!(r, p[2])
        
        step = 250
        mark = floor((P.D[i] + step - 1) / step) * step 
        @constraint(model, z <= mark + 100000 * high[i])
        
        @constraint(model, q[i] >= P.D[i] - z - eps)
        @constraint(model, q[i] >= P.D[i] * high[i] - eps)
    end;

    @objective(model, Min, sum(q .* r))

    optimize!(model)
    
    if return_penalty
        return value.(model[:z]), r
    end
    return value.(model[:z])
end;

end  # module Local
