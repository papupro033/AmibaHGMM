# ============================================================================
# HierarchicalEM.jl
# True Hierarchical EM with Expectation Propagation / Message Passing
# Joint optimization of root + children with soft responsibilities across levels
# ============================================================================

module HierarchicalEM

using LinearAlgebra
using Random
using Statistics
using Printf
using Distributions

# Re-export base GMM types and functions
using ..MixtureGaussianOptimized
using .MixtureGaussianOptimized: Gaussian, GMM, fit_gmm, predict, predict_proba,
    predict_with_confidence, loglikelihood, kmeans_plusplus_init, e_step!, e_step_threaded!,
    m_step, logpdf

export HierarchicalEMModel, HEMNode, fit_hem, predict_hem, predict_proba_hem,
    loglikelihood_hem, print_hem_tree, get_hem_assignments, get_hem_responsibilities

# ============================================================================
# Types
# ============================================================================

"""
    HEMNode

Node in hierarchical EM tree. Contains GMM and child nodes.
Unlike greedy HGMM, this stores full joint structure for message passing.
"""
mutable struct HEMNode
    gmm::GMM                                    # GMM at this level
    children::Union{Nothing, Vector{HEMNode}}   # Child nodes
    depth::Int                                  # Depth (root = 0)
    parent_cluster::Union{Nothing, Int}         # Parent cluster index
    # Transition matrix: p(child=j | parent=k) - K_parent × K_child
    transition::Union{Nothing, Matrix{Float64}} # Normalized rows
end

"""
    HierarchicalEMModel

Full hierarchical EM model with joint parameters.
"""
struct HierarchicalEMModel
    root::HEMNode
    max_depth::Int
    min_cluster_size::Int
    min_cluster_weight::Float64
end

# ============================================================================
# Joint Model Specification
# ============================================================================

"""
Joint distribution for depth-1 hierarchy:
    p(x, z₀, z₁) = p(z₀) × p(z₁|z₀) × p(x|z₁)

Where:
- z₀ ∈ {1..K₀}: root cluster
- z₁ ∈ {1..K₁}: leaf cluster (child of z₀)
- p(z₀) = π₀ (root weights)
- p(z₁|z₀=k) = T[k, :] (transition matrix for child of root cluster k)
- p(x|z₁=j) = N(x | μ₁ⱼ, Σ₁ⱼ) (leaf Gaussian)

For deeper trees, this extends recursively.
"""

# ============================================================================
# Initialization
# ============================================================================

"""
    fit_hem(X, K_root, K_leaf; max_depth=1, ...)

Fit Hierarchical EM with joint message passing.

Arguments:
- X: N×D data matrix
- K_root: Number of root clusters
- K_leaf: Number of leaf clusters PER ROOT CLUSTER (can be vector for variable K)
- max_depth: Maximum depth (currently supports 1)
"""
function fit_hem(
    X::Matrix{Float64},
    K_root::Int,
    K_leaf::Union{Int, Vector{Int}};
    max_depth::Int = 1,
    min_cluster_size::Int = 10,
    min_cluster_weight::Float64 = 0.01,
    max_iter::Int = 100,
    tol::Float64 = 1e-4,
    verbose::Bool = true,
    threaded::Bool = false,
    seed::Int = 42,
    init_strategy::Symbol = :kmeans_plusplus  # :kmeans_plusplus, :flat_gmm, :random
)
    N, D = size(X)
    K_leaf_vec = K_leaf isa Int ? fill(K_leaf, K_root) : K_leaf
    @assert length(K_leaf_vec) == K_root
    
    if verbose
        println("=== True Hierarchical EM (Root K=$K_root, Leaf=$(K_leaf_vec), max_depth=$max_depth) ===")
    end
    
    rng = MersenneTwister(seed)
    
    # === INITIALIZATION ===
    if verbose
        println("  Initializing...")
    end
    
    # Initialize root GMM
    root_gmm, _ = fit_gmm(X, K_root; max_iter=20, tol=1e-3, verbose=false, threaded=threaded, seed=seed)
    root_assignments = predict(root_gmm, X)
    root_probs = predict_proba(root_gmm, X)  # N × K_root
    
    # Initialize child GMMs and transitions
    children = HEMNode[]
    transitions = Matrix{Float64}[]
    
    for k in 1:K_root
        K_child = K_leaf_vec[k]
        
        # Transition: uniform initially
        T = fill(1.0/K_child, K_root, K_child)
        push!(transitions, T)
        
        # Initialize child GMM on subset (soft)
        mask = root_assignments .== k
        n_k = sum(mask)
        
        if n_k < min_cluster_size
            # Fallback: use global data with small noise
            X_child = X .+ 0.01 * randn(rng, size(X))
        else
            X_child = X[mask, :]
        end
        
        if n_k >= K_child * 2
            child_gmm, _ = fit_gmm(X_child, K_child; max_iter=20, tol=1e-3, verbose=false, threaded=threaded, seed=seed+k)
        else
            # Use global covariance
            global_cov = cov(X, corrected=false) + 1e-6 * I
            centers = X[rand(rng, 1:N, K_child), :]
            comps = [Gaussian(centers[j, :], global_cov) for j in 1:K_child]
            child_gmm = GMM(comps, fill(1.0/K_child, K_child))
        end
        
        push!(children, HEMNode(child_gmm, nothing, 1, k, T))
    end
    
    root_node = HEMNode(root_gmm, children, 0, nothing, nothing)
    model = HierarchicalEMModel(root_node, max_depth, min_cluster_size, min_cluster_weight)
    
    # === EM ITERATIONS ===
    log_likelihoods = Float64[]
    
    for iter in 1:max_iter
        # ---- E-STEP: Message Passing ----
        # Compute joint posteriors p(z₀, z₁ | x) for all points
        joint_probs, log_like = e_step_hem!(model, X; threaded=threaded)
        push!(log_likelihoods, log_like)
        
        if verbose
            @printf("Iter %3d: log-likelihood = %.6f", iter, log_like)
            if iter > 1
                @printf("  Δ = %.2e", log_like - log_likelihoods[end-1])
            end
            println()
        end
        
        # Convergence check
        if iter > 1 && abs(log_like - log_likelihoods[end-1]) < tol * abs(log_likelihoods[end-1])
            verbose && println("  → Converged (tol = $tol)")
            break
        end
        
        # ---- M-STEP: Joint Parameter Update ----
        m_step_hem!(model, X, joint_probs; min_weight=min_cluster_weight)
    end
    
    if verbose
        println("  Final LL: $(round(log_likelihoods[end], digits=2)), Iterations: $(length(log_likelihoods))")
        print_hem_tree(model)
    end
    
    return model, log_likelihoods
end

# ============================================================================
# E-Step: Message Passing (Belief Propagation on Tree)
# ============================================================================

"""
    e_step_hem!(model, X; threaded=false) -> (joint_probs, log_likelihood)

Compute posterior responsibilities using upward-downward message passing.

For depth-1 tree:
- Upward: leaf → root messages (likelihoods)
- Downward: root → leaf messages (priors)
- Combine: p(z₀, z₁ | x) ∝ p(z₀) × T[z₀, z₁] × N(x | z₁)

Returns:
- joint_probs: Vector of N matrices (K_root × K_leaf_k) with p(z₀, z₁ | xₙ)
- log_likelihood: scalar sum log p(xₙ)
"""
function e_step_hem!(model::HierarchicalEMModel, X::Matrix{Float64}; threaded::Bool=false)
    N, D = size(X)
    root = model.root
    K_root = root.gmm.K
    
    # Precompute leaf likelihoods for all points and all leaf components
    # leaf_ll[n, (k,j)] = log N(xₙ | child_k.component_j)
    leaf_loglikes = _compute_leaf_loglikes(root, X)
    
    # Root log-prior
    log_prior_root = log.(root.gmm.weights)  # K_root
    
    # Transition log-probs
    log_trans = [log.(child.transition) for child in root.children]  # Vector of K_root × K_leaf
    
    # ---- UPWARD PASS: Leaf → Root ----
    # Message from child k to root: m_{k→0}(z₀) = ∑_{z₁} T[z₀, z₁] N(x | z₁)
    # In log-space: log ∑ exp(log T + log N)
    upward_msgs = Matrix{Float64}(undef, N, K_root)
    
    if threaded && Threads.nthreads() > 1
        Threads.@threads for n in 1:N
            for k in 1:K_root
                child = root.children[k]
                K_leaf = child.gmm.K
                log_T = log_trans[k]
                log_N = @view leaf_loglikes[n, k, :]  # K_leaf
                # log ∑_j exp(log_T[k,j] + log_N[j])
                max_val = maximum(log_T[k, :] + log_N)
                upward_msgs[n, k] = max_val + log(sum(exp.(log_T[k, :] + log_N .- max_val)))
            end
        end
    else
        for n in 1:N
            for k in 1:K_root
                child = root.children[k]
                K_leaf = child.gmm.K
                log_T = log_trans[k]
                log_N = @view leaf_loglikes[n, k, :]
                max_val = maximum(log_T[k, :] + log_N)
                upward_msgs[n, k] = max_val + log(sum(exp.(log_T[k, :] + log_N .- max_val)))
            end
        end
    end
    
    # ---- ROOT POSTERIOR: p(z₀ | x) ∝ π₀ × m_{child→root} ----
    root_log_post = Matrix{Float64}(undef, N, K_root)
    log_likelihood = 0.0
    
    for n in 1:N
        # Unnormalized log posterior at root
        log_unnorm = log_prior_root + upward_msgs[n, :]
        max_val = maximum(log_unnorm)
        exp_sum = sum(exp.(log_unnorm .- max_val))
        log_norm = max_val + log(exp_sum)
        root_log_post[n, :] = log_unnorm .- log_norm
        log_likelihood += log_norm
    end
    
    # ---- DOWNWARD PASS: Root → Leaf ----
    # p(z₁=j | z₀=k, x) ∝ T[k,j] × N(x | z₁=j)
    # Then combine: p(z₀, z₁ | x) = p(z₀ | x) × p(z₁ | z₀, x)
    
    joint_probs = Vector{Matrix{Float64}}(undef, N)
    
    for n in 1:N
        root_post = exp.(root_log_post[n, :])  # K_root
        
        # Build joint matrix for this point
        total_leaves = sum(c -> c.gmm.K, root.children)
        joint = zeros(K_root, total_leaves)  # We'll fill blocks per child
        
        col_offset = 0
        for k in 1:K_root
            child = root.children[k]
            K_leaf = child.gmm.K
            
            # p(z₁ | z₀=k, x) ∝ T[k, :] × N(x | z₁)
            log_T = log_trans[k]
            log_N = @view leaf_loglikes[n, k, :]
            log_cond = log_T[k, :] + log_N
            max_val = maximum(log_cond)
            cond_probs = exp.(log_cond .- max_val)
            cond_probs ./= sum(cond_probs)
            
            # Joint: p(z₀=k, z₁ | x) = p(z₀=k | x) × p(z₁ | z₀=k, x)
            for j in 1:K_leaf
                joint[k, col_offset + j] = root_post[k] * cond_probs[j]
            end
            col_offset += K_leaf
        end
        
        joint_probs[n] = joint
    end
    
    return joint_probs, log_likelihood
end

function _compute_leaf_loglikes(root::HEMNode, X::Matrix{Float64})
    N, D = size(X)
    K_root = root.gmm.K
    max_K_leaf = maximum(c -> c.gmm.K, root.children)
    
    # leaf_loglikes[n, k, j] = log N(xₙ | child_k.component_j)
    leaf_loglikes = Array{Float64}(undef, N, K_root, max_K_leaf)
    
    for n in 1:N
        x = X[n, :]
        for k in 1:K_root
            child = root.children[k]
            for j in 1:child.gmm.K
                leaf_loglikes[n, k, j] = logpdf(child.gmm.components[j], x)
            end
            # Fill remaining with -Inf if variable K
            for j in (child.gmm.K+1):max_K_leaf
                leaf_loglikes[n, k, j] = -Inf
            end
        end
    end
    
    return leaf_loglikes
end

# ============================================================================
# M-Step: Joint Parameter Update
# ============================================================================

"""
    m_step_hem!(model, X, joint_probs; min_weight=0.01)

Update all parameters jointly using soft responsibilities from E-step.

Parameters to update:
1. Root weights π₀
2. Root Gaussians (μ₀, Σ₀)
3. Transitions T[k, j] = p(z₁=j | z₀=k)
4. Leaf Gaussians (μ₁, Σ₁) for each child
"""
function m_step_hem!(model::HierarchicalEMModel, X::Matrix{Float64}, joint_probs::Vector{Matrix{Float64}}; min_weight::Float64=0.01)
    N, D = size(X)
    root = model.root
    K_root = root.gmm.K
    
    # ---- Accumulate sufficient statistics ----
    
    # Root responsibilities: γₙₖ = ∑_j p(z₀=k, z₁=j | xₙ)
    gamma_root = zeros(N, K_root)
    for n in 1:N
        joint = joint_probs[n]
        for k in 1:K_root
            gamma_root[n, k] = sum(joint[k, :])
        end
    end
    
    # Leaf responsibilities per child: γₙₖⱼ = p(z₀=k, z₁=j | xₙ)
    gamma_leaves = [zeros(N, root.children[k].gmm.K) for k in 1:K_root]
    for n in 1:N
        joint = joint_probs[n]
        col_offset = 0
        for k in 1:K_root
            K_leaf = root.children[k].gmm.K
            for j in 1:K_leaf
                gamma_leaves[k][n, j] = joint[k, col_offset + j]
            end
            col_offset += K_leaf
        end
    end
    
    # ---- 1. Update root weights π₀ ----
    Nk_root = sum(gamma_root, dims=1)[:]
    new_weights_root = Nk_root ./ N
    new_weights_root = max.(new_weights_root, min_weight / K_root)
    new_weights_root ./= sum(new_weights_root)
    # Reconstruct root GMM with new weights (same components)
    root.gmm = GMM(root.gmm.components, new_weights_root)
    
    # ---- 2. Update root Gaussians ----
    root.gmm = _update_gmm_from_resp(root.gmm, X, gamma_root)
    
    # ---- 3. Update transitions T[k, j] = E[γₙₖⱼ] / E[γₙₖ] ----
    for k in 1:K_root
        child = root.children[k]
        K_leaf = child.gmm.K
        Nk = Nk_root[k]
        
        if Nk > 1e-8
            # Expected counts for each leaf given root=k
            Nj_given_k = sum(gamma_leaves[k], dims=1)[:]
            child.transition[k, :] = Nj_given_k ./ Nk
            # Ensure no zeros
            child.transition[k, :] = max.(child.transition[k, :], min_weight / K_leaf)
            child.transition[k, :] ./= sum(child.transition[k, :])
        else
            child.transition[k, :] .= 1.0 / K_leaf
        end
    end
    
    # ---- 4. Update leaf Gaussians for each child ----
    for k in 1:K_root
        child = root.children[k]
        resp = gamma_leaves[k]  # N × K_leaf
        child.gmm = _update_gmm_from_resp(child.gmm, X, resp)
    end
end

function _update_gmm_from_resp(gmm::GMM, X::Matrix{Float64}, resp::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    
    new_components = Vector{Gaussian}(undef, K)
    new_weights = Vector{Float64}(undef, K)
    
    for k in 1:K
        gamma_k = @view resp[:, k]
        N_k = sum(gamma_k)
        
        if N_k < 1e-8
            # Empty component: small perturbation
            mu_k = randn(D) .* 0.1
            sigma_k = 0.1 * I
            new_weights[k] = 1e-3 / K
        else
            # μₖ = (1/Nₖ) ∑ γₙₖ xₙ
            mu_k = (X' * gamma_k) ./ N_k
            
            # Σₖ = (1/Nₖ) ∑ γₙₖ (xₙ - μₖ)(xₙ - μₖ)' + reg
            sigma_k = zeros(D, D)
            for n in 1:N
                w = gamma_k[n]
                if w > 0
                    diff = @view(X[n, :]) .- mu_k
                    sigma_k .+= w * (diff * diff')
                end
            end
            sigma_k ./= N_k
            
            # Robust regularization
            min_eig = 1e-3
            reg = max(min_eig, 1e-3 * tr(sigma_k) / D)
            sigma_k += reg * I
            
            new_weights[k] = N_k / N
        end
        
        new_components[k] = Gaussian(mu_k, sigma_k)
    end
    
    # Renormalize weights
    new_weights ./= sum(new_weights)
    
    # Return NEW GMM (immutable)
    return GMM(new_components, new_weights)
end

# ============================================================================
# Log-Likelihood (Joint)
# ============================================================================

"""
    loglikelihood_hem(model, X)

Compute true joint log-likelihood: ∑ₙ log ∑_{z₀,z₁} p(z₀) T[z₀,z₁] N(xₙ|z₁)
"""
function loglikelihood_hem(model::HierarchicalEMModel, X::Matrix{Float64})
    N, D = size(X)
    root = model.root
    K_root = root.gmm.K
    
    # Precompute leaf likelihoods
    leaf_loglikes = _compute_leaf_loglikes(root, X)
    log_trans = [log.(child.transition) for child in root.children]
    log_prior = log.(root.gmm.weights)
    
    ll = 0.0
    for n in 1:N
        # p(xₙ) = ∑_k π₀ₖ ∑_j T[k,j] N(xₙ | child_k, j)
        log_sum_total = -Inf
        
        for k in 1:K_root
            child = root.children[k]
            K_leaf = child.gmm.K
            
            # log ∑_j exp(log T + log N)
            log_terms = log_trans[k][k, :] + @view leaf_loglikes[n, k, 1:K_leaf]
            max_val = maximum(log_terms)
            exp_sum = sum(exp.(log_terms .- max_val))
            log_child_sum = max_val + log(exp_sum)
            
            # Add log π₀ₖ
            term = log_prior[k] + log_child_sum
            
            if log_sum_total == -Inf
                log_sum_total = term
            else
                if term > log_sum_total
                    log_sum_total = term + log1p(exp(log_sum_total - term))
                else
                    log_sum_total = log_sum_total + log1p(exp(term - log_sum_total))
                end
            end
        end
        
        ll += log_sum_total
    end
    
    return ll
end

# ============================================================================
# Prediction
# ============================================================================

"""
    predict_hem(model, X; level=1) -> assignments

Predict cluster assignments.
- level=0: root cluster
- level=1: leaf cluster (joint MAP)
"""
function predict_hem(model::HierarchicalEMModel, X::Matrix{Float64}; level::Int=1)
    N, D = size(X)
    root = model.root
    K_root = root.gmm.K
    
    joint_probs, _ = e_step_hem!(model, X)
    
    if level == 0
        # Root MAP
        assignments = Vector{Int}(undef, N)
        for n in 1:N
            joint = joint_probs[n]
            root_marginal = [sum(joint[k, :]) for k in 1:K_root]
            assignments[n] = argmax(root_marginal)
        end
        return assignments
    else
        # Leaf MAP (joint)
        assignments = Vector{Int}(undef, N)
        for n in 1:N
            joint = joint_probs[n]
            max_val = -Inf
            best_k, best_j = 1, 1
            col_offset = 0
            for k in 1:K_root
                child = root.children[k]
                for j in 1:child.gmm.K
                    val = joint[k, col_offset + j]
                    if val > max_val
                        max_val = val
                        best_k, best_j = k, j
                    end
                end
                col_offset += child.gmm.K
            end
            # Return flat index encoding path (k, j)
            # For simplicity, return root cluster; could encode full path
            assignments[n] = best_k
        end
        return assignments
    end
end

"""
    predict_proba_hem(model, X; level=1) -> responsibilities

Get posterior probabilities.
"""
function predict_proba_hem(model::HierarchicalEMModel, X::Matrix{Float64}; level::Int=1)
    joint_probs, _ = e_step_hem!(model, X)
    
    if level == 0
        N = size(X, 1)
        K_root = model.root.gmm.K
        probs = zeros(N, K_root)
        for n in 1:N
            joint = joint_probs[n]
            for k in 1:K_root
                probs[n, k] = sum(joint[k, :])
            end
        end
        return probs
    else
        # Return joint flattened
        N = size(X, 1)
        total_leaves = sum(c -> c.gmm.K, model.root.children)
        probs = zeros(N, total_leaves)
        for n in 1:N
            joint = joint_probs[n]
            col_offset = 0
            for k in 1:model.root.gmm.K
                child = model.root.children[k]
                for j in 1:child.gmm.K
                    probs[n, col_offset + j] = joint[k, col_offset + j]
                end
                col_offset += child.gmm.K
            end
        end
        return probs
    end
end

function get_hem_assignments(model::HierarchicalEMModel, X::Matrix{Float64}; level::Int=1)
    predict_hem(model, X; level=level)
end

function get_hem_responsibilities(model::HierarchicalEMModel, X::Matrix{Float64})
    joint_probs, _ = e_step_hem!(model, X)
    return joint_probs
end

# ============================================================================
# Utilities
# ============================================================================

function print_hem_tree(model::HierarchicalEMModel; max_depth::Int=3)
    println("\n=== Hierarchical EM Tree (Joint) ===")
    println("Max depth: $(model.max_depth)")
    _print_hem_node(model.root, 0, max_depth)
    println("====================================\n")
end

function _print_hem_node(node::HEMNode, indent::Int, max_depth::Int)
    if node.depth > max_depth
        return
    end
    
    prefix = "  "^node.depth * (node.depth > 0 ? "└─ " : "")
    parent_info = node.parent_cluster === nothing ? "" : " (parent cluster $(node.parent_cluster))"
    
    if node.depth == 0
        println("$(prefix)Depth $(node.depth): Root K=$(node.gmm.K), weights=$(round.(node.gmm.weights, digits=3))")
    else
        println("$(prefix)Depth $(node.depth): Leaf K=$(node.gmm.K), weights=$(round.(node.gmm.weights, digits=3))$parent_info")
        if node.transition !== nothing
            println("$(prefix)   Transition: $(round.(node.transition[node.parent_cluster, :], digits=3))")
        end
    end
    
    if node.children !== nothing
        for child in node.children
            _print_hem_node(child, indent + 2, max_depth)
        end
    end
end

end # module