module Utils

using Clp, JuMP

function evaluate(CHat, CTrue, oracle, objective_fun; loss_only = false)
    n = size(CTrue)[2]
    m = oracle(zeros(size(CTrue)[1]))
    MOI.set(m, MOI.Silent(), true)

    loss = 0
    total = 0
    errors = Vector{Float64}()
    for i=1:size(CHat)[2]
        c_hat = CHat[:,i]
        c_true = CTrue[:,i]

        @objective(m, Min, objective_fun(c_true, m[:w]))
        optimize!(m)

        w_star = value.(m[:w])

        @objective(m, Min, objective_fun(c_hat, m[:w]))
        optimize!(m)

        w_hat = value.(m[:w])

        # if loss_only == false
        #     @objective(m, Min, objective_fun(-c_true, m[:w]))
        #     optimize!(m)
        #
        #     w_worst = value.(m[:w])
        #     z_worst = objective_fun(c_true, w_worst)
        # end

        z_star  = objective_fun(c_true, w_star)
        z_hat   = objective_fun(c_true, w_hat)

        total += z_star
        loss += (z_hat - z_star)

        if loss_only == false
            error = (z_hat - z_star) / z_star
            push!(errors, error)
        end
    end;
    if loss_only
        return loss / total
    else
        return loss / total, errors
    end
end

function evaluate_newsvendor(DHat, DTrue, objective_fun)
    errors = Vector{Float64}()
    for i=1:size(DTrue)[1]
        d_hat = DHat[i]
        d_true = DTrue[i]

        zhat = objective_fun(d_true, d_hat)
        push!(errors, zhat)
    end;
    return errors
end


end
