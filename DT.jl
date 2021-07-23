module DecisionTree_

include("./utils.jl")
import .Utils

using ScikitLearn
@sk_import tree: _tree
@sk_import tree: DecisionTreeRegressor

function train_tree_(X, Y; depth = 4, min_samples_leaf = 4)
    dt = DecisionTreeRegressor(max_depth = depth, min_samples_leaf = min_samples_leaf)
    dt.fit(X, Y)
    return dt
end

function evaluate_newsvendor(X, D, tree, objective_fun)
    pred = tree.predict(X)
    return Utils.evaluate_newsvendor(pred, D', objective_fun)
end

function evaluate_tree_(X, C, tree, oracle, objective_fun; loss_only = false)
    return Utils.evaluate(tree.predict(X)', C', oracle, objective_fun; loss_only = loss_only)
end

end
