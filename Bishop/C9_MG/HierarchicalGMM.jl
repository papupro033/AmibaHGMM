# ============================================================================
# HierarchicalGMM.jl
# Hierarchical Gaussian Mixture Model wrapper around MixtureGaussianOptimized
# Maintains the same API: fit_gmm, predict, predict_proba, predict_with_confidence,
# loglikelihood, kmeans_plusplus_init
# ============================================================================

module HierarchicalGMMs

using LinearAlgebra
using Random
using Statistics
using Printf
using Distributions
# Re-export the base GMM types and functions

using Main.MixtureGaussianOptimized
using Main.MixtureGaussianOptimized: Gaussian, GMM, fit_gmm, predict, predict_proba,
    predict_with_confidence, loglikelihood, kmeans_plusplus_init, e_step!, e_step_threaded!,
    m_step



# ============================================================================
# Hierarchical GMM Types
# ============================================================================

"""
    HGMMNode

A node in the hierarchical GMM tree. Each node contains a GMM and optionally
child nodes for each component (sub-clusters).
"""
mutable struct HGMMNode
    gmm::GMM                                    # GMM at this level
    children::Union{Nothing, Vector{HGMMNode}}  # Child HGMMNodes (one per component)
    depth::Int                                  # Depth in tree (root = 0)
    parent_cluster::Union{Nothing, Int}         # Which parent cluster this node belongs to
end

"""
    HierarchicalGMM

A hierarchical GMM model consisting of a root node and tree of sub-clusters.
"""
struct HierarchicalGMM
    root::HGMMNode
    max_depth::Int
    min_cluster_size::Int
    min_cluster_weight::Float64
    split_criterion::Symbol  # :bic, :aic, :ll_threshold, :fixed_depth
    split_threshold::Float64
end

# ============================================================================
# Hierarchical GMM Construction
# ============================================================================

"""
    fit_hgmm(X, K; max_depth=2, min_cluster_size=10, min_cluster_weight=0.01,
              split_criterion=:fixed_depth, split_threshold=0.0,
              max_iter=100, tol=1e-4, verbose=true, threaded=false, seed=42)

Fit a Hierarchical GMM to data X.

Arguments:
- X: N×D data matrix
- K: Number of top-level clusters
- max_depth: Maximum depth of hierarchy (0 = flat GMM)
- min_cluster_size: Minimum points in a cluster to allow splitting
- min_cluster_weight: Minimum weight for a cluster to allow splitting
- split_criterion: Criterion for splitting (:fixed_depth, :bic, :aic, :ll_threshold)
- split_threshold: Threshold for BIC/AIC/LL improvement to justify split
- max_iter, tol, verbose, threaded, seed: Passed to fit_gmm

Returns:
- HierarchicalGMM object
"""
function fit_hgmm(
    X::Matrix{Float64},
    K::Int;
    max_depth::Int = 2,
    min_cluster_size::Int = 10,
    min_cluster_weight::Float64 = 0.01,
    split_criterion::Symbol = :fixed_depth,
    split_threshold::Float64 = 0.0,
    max_iter::Int = 100,
    tol::Float64 = 1e-4,
    verbose::Bool = true,
    threaded::Bool = false,
    seed::Int = 42
)
    if verbose
        println("=== Hierarchical GMM (K=$K, max_depth=$max_depth, criterion=$split_criterion) ===")
    end
    
    # Fit root GMM
    if verbose
        println("  Depth 0: Fitting root GMM with K=$K...")
    end
    root_gmm, history = fit_gmm(X, K; max_iter=max_iter, tol=tol, verbose=verbose, threaded=threaded, seed=seed)
    
    # Get assignments for root
    root_assignments = predict(root_gmm, X)
    root_probs = predict_proba(root_gmm, X)
    
    # Build hierarchy recursively
    root_node = HGMMNode(root_gmm, nothing, 0, nothing)
    
    if max_depth > 0
        _build_hierarchy!(
            root_node, X, root_assignments, root_probs;
            max_depth=max_depth,
            min_cluster_size=min_cluster_size,
            min_cluster_weight=min_cluster_weight,
            split_criterion=split_criterion,
            split_threshold=split_threshold,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            threaded=threaded,
            seed=seed
        )
    end
    
    hgmm = HierarchicalGMM(
        root_node,
        max_depth,
        min_cluster_size,
        min_cluster_weight,
        split_criterion,
        split_threshold
    )
    
    if verbose
        println("  Hierarchy built. Total nodes: $(count_nodes(hgmm.root))")
        print_hgmm_tree(hgmm)
    end
    
    return hgmm
end

# ============================================================================
# Hierarchy Building (Internal)
# ============================================================================

function _build_hierarchy!(
    node::HGMMNode,
    X::Matrix{Float64},
    parent_assignments::Vector{Int},
    parent_probs::Matrix{Float64};
    max_depth::Int,
    min_cluster_size::Int,
    min_cluster_weight::Float64,
    split_criterion::Symbol,
    split_threshold::Float64,
    max_iter::Int,
    tol::Float64,
    verbose::Bool,
    threaded::Bool,
    seed::Int
)
    if node.depth >= max_depth
        return
    end
    
    K = node.gmm.K
    children = HGMMNode[]
    
    for k in 1:K
        # Get points assigned to this cluster (hard assignment)
        cluster_mask = parent_assignments .== k
        n_cluster = sum(cluster_mask)
        
        # Check minimum cluster size
        if n_cluster < min_cluster_size
            if verbose && node.depth < max_depth - 1
                println("    Depth $(node.depth+1), Cluster $k: Skipping split (n=$n_cluster < $min_cluster_size)")
            end
            continue
        end
        
        # Check minimum weight
        if node.gmm.weights[k] < min_cluster_weight
            if verbose && node.depth < max_depth - 1
                println("    Depth $(node.depth+1), Cluster $k: Skipping split (weight=$(round(node.gmm.weights[k],4)) < $min_cluster_weight)")
            end
            continue
        end
        
        # Extract cluster data
        X_cluster = X[cluster_mask, :]
        
        # Determine number of sub-clusters
        if split_criterion == :fixed_depth
            n_subclusters = 2  # Default binary split
        elseif split_criterion in (:bic, :aic)
            n_subclusters = _select_n_subclusters_bic(X_cluster; max_k=4, verbose=false)
        else
            n_subclusters = 2
        end
        
        if verbose
            println("    Depth $(node.depth+1), Parent Cluster $k: Fitting sub-GMM (K=$n_subclusters, n=$n_cluster)...")
        end
        
        # Fit sub-GMM
        sub_gmm, sub_history = fit_gmm(
            X_cluster, n_subclusters;
            max_iter=max_iter, tol=tol, verbose=verbose && node.depth < 1,
            threaded=threaded, seed=seed + k + node.depth * 100
        )
        
        # Get sub-assignments
        sub_assignments = predict(sub_gmm, X_cluster)
        sub_probs = predict_proba(sub_gmm, X_cluster)
        
        # Create child node
        child_node = HGMMNode(sub_gmm, nothing, node.depth + 1, k)
        
        # Recursively build deeper hierarchy
        if node.depth + 1 < max_depth
            # Create global assignments for this subcluster
            global_assignments = copy(parent_assignments)
            global_assignments[cluster_mask] = sub_assignments .+ maximum(parent_assignments)
            
            # Create global probabilities (approximate)
            global_probs = copy(parent_probs)
            global_probs[cluster_mask, :] = sub_probs
            
            _build_hierarchy!(
                child_node, X, global_assignments, global_probs;
                max_depth=max_depth,
                min_cluster_size=min_cluster_size,
                min_cluster_weight=min_cluster_weight,
                split_criterion=split_criterion,
                split_threshold=split_threshold,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                threaded=threaded,
                seed=seed
            )
        end
        
        push!(children, child_node)
    end
    
    if !isempty(children)
        node.children = children
    end
end

function _select_n_subclusters_bic(X::Matrix{Float64}; max_k::Int=4, verbose::Bool=false)
    best_k = 2
    best_bic = Inf
    
    for k in 2:max_k
        gmm, _ = fit_gmm(X, k; max_iter=50, verbose=false)
        n, d = size(X)
        ll = loglikelihood(gmm, X)
        n_params = k * (d + d*(d+1)/2 + 1) - 1  # weights + means + covs - 1 constraint
        bic = -2 * ll + n_params * log(n)
        
        if verbose
            println("      K=$k: LL=$(round(ll,2)), BIC=$(round(bic,2))")
        end
        
        if bic < best_bic
            best_bic = bic
            best_k = k
        end
    end
    
    return best_k
end

# ============================================================================
# ============================================================================
# Helper: compute log-likelihood scores for a single point
# ============================================================================
function _compute_logprobs(node::HGMMNode, x::Vector{Float64})
   scores = Float64[]
   for (c, w) in zip(node.gmm.components, node.gmm.weights)
       push!(scores, MixtureGaussianOptimized.logpdf(c, x) + log(w))
   end
   return scores
end

# ============================================================================
# Hierarchical GMM Prediction
# ============================================================================

"""
    predict_hgmm(hgmm::HierarchicalGMM, X::Matrix{Float64}; 
                  level::Int=0, return_path::Bool=false)

Predict cluster assignments using hierarchical GMM.

Arguments:
- hgmm: HierarchicalGMM model
- X: N×D data matrix
- level: Depth level to predict at (0 = root only, max_depth = deepest)
- return_path: If true, return full path from root to leaf

Returns:
- If level=0 or return_path=false: Vector{Int} of assignments at requested level
- If return_path=true: Vector{Vector{Int}} where each inner vector is [root_cluster, sub_cluster, ...]
"""
function predict_hgmm(
    hgmm::HierarchicalGMM,
    X::Matrix{Float64};
    level::Int = hgmm.max_depth,
    return_path::Bool = false
)
    N, D = size(X)
    level = min(level, hgmm.max_depth)
    
    # Root level predictions
    root_assignments = predict(hgmm.root.gmm, X)
    
    if level == 0 || hgmm.root.children === nothing
        return return_path ? [[a] for a in root_assignments] : root_assignments
    end
    
    # For deeper levels, traverse the tree
    if return_path
        paths = Vector{Vector{Int}}(undef, N)
        for i in 1:N
            paths[i] = _predict_path(hgmm.root, X[i, :], level)
        end
        return paths
    else
        assignments = Vector{Int}(undef, N)
        for i in 1:N
            assignments[i] = _predict_at_level(hgmm.root, X[i, :], level)
        end
        return assignments
    end
end

function _predict_path(node::HGMMNode, x::Vector{Float64}, target_level::Int)
    # Root assignment
    scores = _compute_logprobs(node, x)
    root_k = argmax(scores)
    path = [root_k]
    
    if target_level == 0 || node.children === nothing
        return path
    end
    
    # Traverse to child
    if root_k <= length(node.children)
        child = node.children[root_k]
        child_path = _predict_path(child, x, target_level - 1)
        append!(path, child_path)
    end
    
    return path
end

function _predict_at_level(node::HGMMNode, x::Vector{Float64}, target_level::Int)
    if target_level == 0 || node.children === nothing
        scores = _compute_logprobs(node, x)
        return argmax(scores)
    end
    
    scores = _compute_logprobs(node, x)
    root_k = argmax(scores)
    
    if root_k <= length(node.children)
        return _predict_at_level(node.children[root_k], x, target_level - 1)
    else
        return root_k
    end
end

"""
    predict_proba_hgmm(hgmm::HierarchicalGMM, X::Matrix{Float64}; level::Int=0)

Get posterior probabilities at a specific hierarchy level.
"""
function predict_proba_hgmm(
    hgmm::HierarchicalGMM,
    X::Matrix{Float64};
    level::Int = 0
)
    if level == 0 || hgmm.root.children === nothing
        return predict_proba(hgmm.root.gmm, X)
    end
    
    # For deeper levels, compute joint probabilities along paths
    N, D = size(X)
    K_root = hgmm.root.gmm.K
    
    # Get root probabilities
    root_probs = predict_proba(hgmm.root.gmm, X)
    
    if level == 1 && hgmm.root.children !== nothing
        # Compute joint probabilities for level 1
        total_subclusters = sum(c -> c.gmm.K, hgmm.root.children)
        joint_probs = zeros(N, total_subclusters)
        
        col_offset = 0
        for (k, child) in enumerate(hgmm.root.children)
            K_child = child.gmm.K
            child_probs = predict_proba(child.gmm, X)
            for n in 1:N
                for j in 1:K_child
                    joint_probs[n, col_offset + j] = root_probs[n, k] * child_probs[n, j]
                end
            end
            col_offset += K_child
        end
        
        # Normalize
        row_sums = sum(joint_probs, dims=2)
        joint_probs ./= row_sums
        return joint_probs
    end
    
    return root_probs
end

"""
    predict_with_confidence_hgmm(hgmm::HierarchicalGMM, X::Matrix{Float64}; level::Int=0)

Predict with confidence scores at specified hierarchy level.
"""
function predict_with_confidence_hgmm(
    hgmm::HierarchicalGMM,
    X::Matrix{Float64};
    level::Int = 0
)
    if level == 0 || hgmm.root.children === nothing
        return predict_with_confidence(hgmm.root.gmm, X)
    end
    
    assignments = predict_hgmm(hgmm, X; level=level)
    probs = predict_proba_hgmm(hgmm, X; level=level)
    
    N = size(X, 1)
    confidences = Vector{Float64}(undef, N)
    
    for i in 1:N
        confidences[i] = probs[i, assignments[i]] * 100.0
    end
    
    return assignments, confidences
end

"""
    loglikelihood_hgmm(hgmm::HierarchicalGMM, X::Matrix{Float64})

Compute total log-likelihood of data under hierarchical GMM.
"""
function loglikelihood_hgmm(hgmm::HierarchicalGMM, X::Matrix{Float64})
    return _loglikelihood_hgmm_node(hgmm.root, X)
end

function _loglikelihood_hgmm_node(node::HGMMNode, X::Matrix{Float64})
    # Log-likelihood at this node
    ll = loglikelihood(node.gmm, X)
    
    # Add children log-likelihoods (weighted by parent responsibilities)
    if node.children !== nothing
        resp = predict_proba(node.gmm, X)
        for (k, child) in enumerate(node.children)
            if k <= size(resp, 2)
                weights = resp[:, k]
                if sum(weights) > 1e-10
                    ll += sum(weights .* _loglikelihood_hgmm_node_per_point(child, X))
                end
            end
        end
    end
    
    return ll
end

function _loglikelihood_hgmm_node_per_point(node::HGMMNode, X::Matrix{Float64})
    N = size(X, 1)
    ll_per_point = Vector{Float64}(undef, N)
    for i in 1:N
        ll_per_point[i] = loglikelihood(node.gmm, X[i:i, :])
    end
    return ll_per_point
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    count_nodes(node::HGMMNode)

Count total nodes in HGMM tree.
"""
function count_nodes(node::HGMMNode)
    count = 1
    if node.children !== nothing
        for child in node.children
            count += count_nodes(child)
        end
    end
    return count
end

count_nodes(hgmm::HierarchicalGMM) = count_nodes(hgmm.root)

"""
    get_hgmm_assignments(hgmm::HierarchicalGMM, X::Matrix{Float64}; level::Int=0)

Get flat assignments at a specific hierarchy level (convenience wrapper).
"""
get_hgmm_assignments(hgmm::HierarchicalGMM, X::Matrix{Float64}; level::Int=0) = 
    predict_hgmm(hgmm, X; level=level)

"""
    get_hgmm_flat_assignments(hgmm::HierarchicalGMM, X::Matrix{Float64})

Get flat assignments at the deepest level (each leaf cluster gets unique ID).
"""
function get_hgmm_flat_assignments(hgmm::HierarchicalGMM, X::Matrix{Float64})
    paths = predict_hgmm(hgmm, X; return_path=true)
    flat_assignments = Vector{Int}(undef, length(paths))
    
    # Map each unique path to a flat cluster ID
    path_to_id = Dict{Vector{Int}, Int}()
    next_id = 1
    
    for (i, path) in enumerate(paths)
        if !haskey(path_to_id, path)
            path_to_id[path] = next_id
            next_id += 1
        end
        flat_assignments[i] = path_to_id[path]
    end
    
    return flat_assignments, path_to_id
end

"""
    print_hgmm_tree(hgmm::HierarchicalGMM; max_depth::Int=3)

Print hierarchical GMM tree structure.
"""
function print_hgmm_tree(hgmm::HierarchicalGMM; max_depth::Int=3)
    println("\n=== Hierarchical GMM Tree ===")
    println("Max depth: $(hgmm.max_depth), Nodes: $(count_nodes(hgmm))")
    println("Split criterion: $(hgmm.split_criterion), Threshold: $(hgmm.split_threshold)")
    _print_node(hgmm.root, 0, max_depth)
    println("==============================\n")
end

function _print_node(node::HGMMNode, indent::Int, max_depth::Int)
    if node.depth > max_depth
        return
    end
    
    prefix = "  "^node.depth * (node.depth > 0 ? "└─ " : "")
    parent_info = node.parent_cluster === nothing ? "" : " (parent cluster $(node.parent_cluster))"
    println("$(prefix)Depth $(node.depth): K=$(node.gmm.K), weights=$(round.(node.gmm.weights, digits=3))$parent_info")
    
    if node.children !== nothing
        for child in node.children
            _print_node(child, indent + 2, max_depth)
        end
    end
end

"""
    get_hgmm_stats(hgmm::HierarchicalGMM, X::Matrix{Float64}; level::Int=0)

Get statistics for clusters at a specific hierarchy level.
"""
function get_hgmm_stats(hgmm::HierarchicalGMM, X::Matrix{Float64}; level::Int=0)
    assignments = predict_hgmm(hgmm, X; level=level)
    K_eff = maximum(assignments)
    
    stats = Dict{Int, Dict}()
    for k in 1:K_eff
        mask = assignments .== k
        stats[k] = Dict(
            :size => sum(mask),
            :proportion => sum(mask) / length(assignments),
            :mean => mean(X[mask, :], dims=1)[:],
            :cov => cov(X[mask, :], corrected=false)
        )
    end
    
    return stats
end

export HierarchicalGMM, HGMMNode, fit_hgmm, predict_hgmm, predict_proba_hgmm,
  predict_with_confidence_hgmm, loglikelihood_hgmm, print_hgmm_tree,
  get_hgmm_assignments, get_hgmm_flat_assignments, get_hgmm_stats
export count_nodes
end # module


#data=hstack(vec(red.(img)),vec(green.(img)),vec(blue.(img)))
t=length(res1)
res2=Vector{Int}(undef,t)
for (idx,i) in enumerate(res1)
res2[idx]=sum(i)
end
