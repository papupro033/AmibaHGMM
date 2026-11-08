

module HGMM_m

using LinearAlgebra
using Random
using Statistics
using Printf

# ---------------------------------------------------------------------------
# Definiciones previas: Gaussian, GMM, TreeTopology (Sin cambios)
# ---------------------------------------------------------------------------
struct Gaussian
    mu::Vector{Float64}
    sigma::Matrix{Float64}
    precision::Matrix{Float64}
    logdet::Float64
    norm_const::Float64
    dim::Int
end

function Gaussian(mu::AbstractVector, sigma::AbstractMatrix; reg::Float64=1e-6)
    d = length(mu)
    sigma_reg = sigma + reg * I
    F = cholesky(Symmetric(sigma_reg); check=false)
    if F.info != 0
        sigma_reg = sigma + 1e-3 * I
        F = cholesky(Symmetric(sigma_reg); check=false)
        if F.info != 0
            error("Cholesky failed even after regularization")
        end
    end
    precision = F.U' \ (F.U \ I)
    logdet = 2.0 * sum(log.(diag(F.U)))
    norm_const = -0.5 * d * log(2π) - 0.5 * logdet
    return Gaussian(mu, sigma_reg, precision, logdet, norm_const, d)
end

function Gaussian(mu::Real, sigma::Real; reg::Float64=1e-6)
    sigma_reg = sigma + reg
    precision = 1.0 / sigma_reg
    logdet = log(sigma_reg)
    norm_const = -0.5 * log(2π) - 0.5 * logdet
    return Gaussian([mu], [sigma_reg], [precision], logdet, norm_const, 1)
end

@inline function logpdf(g::Gaussian, x::AbstractVector)
    diff = x .- g.mu
    mahal = dot(diff, g.precision * diff)
    return g.norm_const - 0.5 * mahal
end

@inline function logpdf(g::Gaussian, x::Real)
    diff = x - g.mu[1]
    return g.norm_const - 0.5 * g.precision[1] * diff * diff
end

struct GMM
    components::Vector{Gaussian}
    weights::Vector{Float64}
    log_weights::Vector{Float64}
    K::Int
end

function GMM(components::Vector{Gaussian}, weights::AbstractVector)
    logw = log.(weights)
    return GMM(components, weights, logw, length(components))
end

struct TreeTopology
    parents::Vector{Int}
    children::Vector{Vector{Int}}
    levels::Vector{Vector{Int}}
end
# se meete un vector donde solo indicas cual es el padre del nodo y por consecuencia la longitud del vector es el numero de nodos
function TreeTopology(parents::Vector{Int})
    n_nodes = length(parents)
    children = [Int[] for _ in 1:n_nodes]
    for (child_id, parent_id) in enumerate(parents)
        if parent_id > 0
            push!(children[parent_id], child_id)
        end
    end
    levels = Vector{Int}[]
    current_level = [findfirst(==(0), parents)]
    while !isempty(current_level)
        push!(levels, current_level)
        next_level = Int[]
        for node in current_level
            append!(next_level, children[node])
        end
        current_level = next_level
    end
    return TreeTopology(parents, children, levels)
end

# ---------------------------------------------------------------------------
# NUEVA ESTRUCTURA: HGMM
# ---------------------------------------------------------------------------
struct HGMM
    topology::TreeTopology
    gmms::Vector{GMM}
end

# ---------------------------------------------------------------------------
# Funciones EM Base (Sin cambios)
# ---------------------------------------------------------------------------
function e_step!(log_gamma::Matrix{Float64}, gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    log_like = 0.0
    @inbounds for n in 1:N
        x = @view X[n, :]
        log_num = Vector{Float64}(undef, K)
        for k in 1:K
            log_num[k] = gmm.log_weights[k] + logpdf(gmm.components[k], x)
        end
        max_log = maximum(log_num)
        exp_sum = 0.0
        for k in 1:K
            exp_sum += exp(log_num[k] - max_log)
        end
        log_sum = max_log + log(exp_sum)
        for k in 1:K
            log_gamma[n, k] = log_num[k] - log_sum
        end
        log_like += log_sum
    end
    return log_like
end

function e_step_threaded!(log_gamma::Matrix{Float64}, gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    log_like = Threads.Atomic{Float64}(0.0)
    Threads.@threads for n in 1:N
        x = @view X[n, :]
        log_num = Vector{Float64}(undef, K)
        for k in 1:K
            log_num[k] = gmm.log_weights[k] + logpdf(gmm.components[k], x)
        end
        max_log = maximum(log_num)
        exp_sum = 0.0
        for k in 1:K
            exp_sum += exp(log_num[k] - max_log)
        end
        log_sum = max_log + log(exp_sum)
        for k in 1:K
            log_gamma[n, k] = log_num[k] - log_sum
        end
        Threads.atomic_add!(log_like, log_sum)
    end
    return log_like[]
end

function m_step(gmm::GMM, X::Matrix{Float64}, log_gamma::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    new_components = Vector{Gaussian}(undef, K)
    new_weights = Vector{Float64}(undef, K)
    for k in 1:K
        gamma_k = exp.(@view log_gamma[:, k])
        N_k = sum(gamma_k)
        mu_k = zeros(D)
        @inbounds for n in 1:N
            w = gamma_k[n]
            if w > 0
                mu_k .+= w * @view X[n, :]
            end
        end
        mu_k ./= N_k
        sigma_k = zeros(D, D)
        @inbounds for n in 1:N
            w = gamma_k[n]
            if w > 0
                diff = @view(X[n, :]) .- mu_k
                sigma_k .+= w * (diff * diff')
            end
        end
        sigma_k ./= N_k
        min_eig = 1e-3
        reg = max(min_eig, 1e-3 * tr(sigma_k) / D)
        sigma_k += reg * I
        min_weight = 1e-3 / K
        new_weights[k] = max(N_k / N, min_weight)
        new_components[k] = Gaussian(mu_k, sigma_k)
    end
    new_weights ./= sum(new_weights)
    return GMM(new_components, new_weights)
end

function kmeans_plusplus_init(X::Matrix{Float64}, K::Int; rng::AbstractRNG=Random.default_rng())
    N, D = size(X)
    centers = Matrix{Float64}(undef, K, D)
    centers[1, :] = X[rand(rng, 1:N), :]
    dist2 = fill(Inf, N)
    for k in 2:K
        @inbounds for n in 1:N
            d2 = sum(abs2, X[n, :] - centers[k-1, :])
            if d2 < dist2[n]
                dist2[n] = d2
            end
        end
        probs = dist2 ./ sum(dist2)
        idx = findfirst(cumsum(probs) .>= rand(rng))
        centers[k, :] = X[idx, :]
    end
    global_cov = cov(X, corrected=false) + 1e-6 * I
    components = [Gaussian(centers[k, :], global_cov) for k in 1:K]
    weights = fill(1.0/K, K)
    return GMM(components, weights)
end

function fit_gmm(
    X::Matrix{Float64},
    K::Int;
    max_iter::Int=100,
    tol::Float64=1e-4,
    verbose::Bool=true,
    threaded::Bool=false,
    seed::Int=42
)
    rng = MersenneTwister(seed)
    N, D = size(X)
    gmm = kmeans_plusplus_init(X, K; rng=rng)
    log_gamma = Matrix{Float64}(undef, N, K)
    log_likelihoods = Float64[]
    log_like_prev = -Inf
    for iter in 1:max_iter
        if threaded && Threads.nthreads() > 1
            log_like = e_step_threaded!(log_gamma, gmm, X)
        else
            log_like = e_step!(log_gamma, gmm, X)
        end
        push!(log_likelihoods, log_like)
        if verbose
            @printf("Iter %3d: log-likelihood = %.6f", iter, log_like)
            if iter > 1
                @printf("  Δ = %.2e", log_like - log_like_prev)
            end
            println()
        end
        if iter > 1 && abs(log_like - log_like_prev) < tol * abs(log_like_prev)
            verbose && println("  → Convergido (tol = $tol)")
            break
        end
        log_like_prev = log_like
        gmm = m_step(gmm, X, log_gamma)
    end
    return gmm, log_likelihoods
end

# ---------------------------------------------------------------------------
# fit_gmm PARA MODELOS JERÁRQUICOS (NUEVO)
# ---------------------------------------------------------------------------
function fit_gmm(
    X::Matrix{Float64},
    topology::TreeTopology;
    leafs::Union{Int,Vector{Int}}=2, # ← Corregido la coma faltante aquí
    max_iter::Int=100,
    tol::Float64=1e-4,
    verbose::Bool=true,
    threaded::Bool=false,
    seed::Int=42
)
    N, D = size(X)
    n_nodes = length(topology.parents)
    leaf_nodes = topology.levels[end]
    punt = length(leaf_nodes)
    
    # Manejo de los componentes de las hojas
    if typeof(leafs) <: Int
        puntas = fill(leafs, punt)
    else 
        if length(leafs) == punt
            puntas = leafs
        else
            error("No tiene el mismo numero de componentes que hojas")
        end
    end

    gmms = Vector{GMM}(undef, n_nodes)
    
    # Manejo de Datos: Guardar qué índices (filas de X) pertenecen a qué nodo
    data_indices = [Int[] for _ in 1:n_nodes]
    
    # Asignar todo el dataset a la raíz (asumiendo que el nivel 1 tiene un solo nodo, la raíz)
    root_node = topology.levels[1][1]
    data_indices[root_node] = collect(1:N)
    
    log_likelihoods_dict = Dict{Int, Vector{Float64}}()

    # Recorrido del árbol por niveles
    for (level_idx, nivel) in enumerate(topology.levels)
        is_leaf_level = (level_idx == length(topology.levels))
        
        for (idx_in_level, node_id) in enumerate(nivel)
            X_node = X[data_indices[node_id], :]
            N_node = size(X_node, 1)
            
            if N_node == 0
                @warn "El nodo $node_id no tiene datos asignados."
                continue
            end
            
            # Determinar la K para este nodo (hijos o componentes finales)
            if is_leaf_level
                K_node = puntas[idx_in_level]
            else
                K_node = length(topology.children[node_id])
            end
            
            # Entrenar el GMM con la función general
            verbose && println("Ajustando nodo $node_id con $K_node componentes y $N_node muestras...")
            gmm, lls = fit_gmm(X_node, K_node; max_iter=max_iter, tol=tol, verbose=verbose, threaded=threaded, seed=seed)
            gmms[node_id] = gmm
            log_likelihoods_dict[node_id] = lls
            
            # SPLIT DE DATOS PARA LOS HIJOS:
            if !is_leaf_level

                # Usamos la función predict(gmm, X) definida abajo
                assignments = predict(gmm, X_node)
                children_ids = topology.children[node_id]
                
                # Enviar cada índice local al hijo correspondiente
                for i in 1:N_node
                    child_idx = assignments[i]
                    child_node_id = children_ids[child_idx]
                    push!(data_indices[child_node_id], data_indices[node_id][i])
                end
            end
        end
    end
    
    return HGMM(topology, gmms), log_likelihoods_dict, data_indices
end

# ---------------------------------------------------------------------------
# PREDICT PARA GMM NORMAL Y PARA HGMM
# ---------------------------------------------------------------------------
function predict(gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    assignments = Vector{Int}(undef, N)
    for n in 1:N
        x = @view X[n, :]
        best_k = 1
        best_logprob = -Inf
        for k in 1:K
            lp = gmm.log_weights[k] + logpdf(gmm.components[k], x)
            if lp > best_logprob
                best_logprob = lp
                best_k = k
            end
        end
        assignments[n] = best_k
    end
    return assignments
end

"""
    predict(hgmm::HGMM, X::Matrix{Float64}) -> (leaf_assignments, component_assignments)

Navega el árbol prediciendo iterativamente de padre a hijo hasta llegar a una hoja.
Retorna una tupla de arreglos: el ID de la hoja asignada, y el componente final asignado dentro de esa hoja.
"""
function predict(hgmm::HGMM, X::Matrix{Float64})
    N, D = size(X)
    leaf_assignments = Vector{Int}(undef, N)
    comp_assignments = Vector{Int}(undef, N)
    
    for n in 1:N
        x = @view X[n, :]
        
        # Empezar desde la raíz
        current_node = hgmm.topology.levels[1][1]
        
        # Bajar por el árbol hasta topar con un nodo sin hijos (hoja)
        while !isempty(hgmm.topology.children[current_node])
            gmm = hgmm.gmms[current_node]
            best_k = argmax([gmm.log_weights[k] + logpdf(gmm.components[k], x) for k in 1:gmm.K])
            # El componente nos dice a qué nodo hijo avanzar
            current_node = hgmm.topology.children[current_node][best_k]
        end
        
        # Una vez en la hoja, predecir el componente interno
        leaf_gmm = hgmm.gmms[current_node]
        best_k_leaf = argmax([leaf_gmm.log_weights[k] + logpdf(leaf_gmm.components[k], x) for k in 1:leaf_gmm.K])
        
        leaf_assignments[n] = current_node
        comp_assignments[n] = best_k_leaf
    end
    
    return leaf_assignments, comp_assignments
end

# ---------------------------------------------------------------------------
# LOGLIKELIHOOD PARA GMM NORMAL Y PARA HGMM
# ---------------------------------------------------------------------------
function loglikelihood(gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    ll = 0.0
    for n in 1:N
        x = @view X[n, :]
        log_sum = -Inf
        for k in 1:gmm.K
            lp = gmm.log_weights[k] + logpdf(gmm.components[k], x)
            if log_sum == -Inf
                log_sum = lp
            else
                log_sum = lp > log_sum ? lp + log1p(exp(log_sum - lp)) : log_sum + log1p(exp(lp - log_sum))
            end
        end
        ll += log_sum
    end
    return ll
end

function predict_niveles(hgmm::HGMM, X::Matrix{Float64})
    N, D = size(X)
    max_niveles = length(hgmm.topology.levels)
    
    # Matriz para guardar el nodo asignado en cada nivel para cada pixel
    # dimensiones: (max_niveles, N)
    asignaciones_nivel = zeros(Int, max_niveles, N)
    comp_assignments = Vector{Int}(undef, N)
    
    for n in 1:N
        x = @view X[n, :]
        
        # Empezar desde la raíz (Nivel 1)
        current_node = hgmm.topology.levels[1][1]
        asignaciones_nivel[1, n] = current_node
        
        # Bajar por el resto de los niveles
        for l in 2:max_niveles
            # Si el nodo actual tiene hijos, calculamos a cuál debe bajar
            if !isempty(hgmm.topology.children[current_node])
                gmm = hgmm.gmms[current_node]
                best_k = argmax([gmm.log_weights[k] + logpdf(gmm.components[k], x) for k in 1:gmm.K])
                current_node = hgmm.topology.children[current_node][best_k]
            end
            
            # Guardamos el nodo actual para este nivel. 
            # (Si ya era una hoja, simplemente se repite el mismo nodo en niveles más profundos)
            asignaciones_nivel[l, n] = current_node
        end
        
        # Componente interno de la hoja final
        leaf_gmm = hgmm.gmms[current_node]
        best_k_leaf = argmax([leaf_gmm.log_weights[k] + logpdf(leaf_gmm.components[k], x) for k in 1:leaf_gmm.K])
        comp_assignments[n] = best_k_leaf
    end
    
    return asignaciones_nivel, comp_assignments
end

"""
    loglikelihood(hgmm::HGMM, X::Matrix{Float64})

Calcula el log-likelihood total del modelo jerárquico. Obtiene todos los caminos 
hacia las hojas para generar la distribución conjunta global probabilística.
"""
function loglikelihood(hgmm::HGMM, X::Matrix{Float64})
    N, D = size(X)
    
    # Recolectar recursivamente todos los componentes base y la suma de sus log-pesos de ruteo
    global_components = Gaussian[]
    global_log_weights = Float64[]
    
    function extract_leaves(node::Int, accumulated_log_weight::Float64)
        gmm = hgmm.gmms[node]
        if isempty(hgmm.topology.children[node]) # Es nodo hoja
            for k in 1:gmm.K
                push!(global_components, gmm.components[k])
                push!(global_log_weights, accumulated_log_weight + gmm.log_weights[k])
            end
        else
            for (k, child) in enumerate(hgmm.topology.children[node])
                extract_leaves(child, accumulated_log_weight + gmm.log_weights[k])
            end
        end
    end
    
    root_node = hgmm.topology.levels[1][1]
    extract_leaves(root_node, 0.0)
    
    ll = 0.0
    for n in 1:N
        x = @view X[n, :]
        log_sum = -Inf
        for i in 1:length(global_components)
            lp = global_log_weights[i] + logpdf(global_components[i], x)
            if log_sum == -Inf
                log_sum = lp
            else
                log_sum = lp > log_sum ? lp + log1p(exp(log_sum - lp)) : log_sum + log1p(exp(lp - log_sum))
            end
        end
        ll += log_sum
    end
    return ll
end


export Gaussian, GMM, fit_gmm, predict, predict_proba, predict_with_confidence, loglikelihood, kmeans_plusplus_init, predict_niveles

export HGMM, TreeTopology

##Visulaizacion
"""
    get_means(gmm::GMM) -> Matrix{Float64}

Devuelve una matriz de tamaño (K × D) con las medias de los componentes del GMM.
Ideal para mapear las clases asignadas a sus valores (colores) representativos.
"""
function get_means(gmm::GMM)
    D = gmm.components[1].dim
    K = gmm.K
    means = Matrix{Float64}(undef, K, D)
    
    for k in 1:K
        means[k, :] = gmm.components[k].mu
    end
    
    return means
end

"""
    get_means(hgmm::HGMM) -> Dict{Tuple{Int, Int}, Vector{Float64}}

Extrae las medias de todos los componentes en los **nodos hoja** del HGMM.
Devuelve un diccionario donde la llave es (ID_nodo_hoja, ID_componente) y 
el valor es el vector medio (μ).
"""
function get_means(hgmm::HGMM)
    means_dict = Dict{Tuple{Int, Int}, Vector{Float64}}()
    n_nodes = length(hgmm.topology.parents)
    
    for node in 1:n_nodes
        # Verificamos si es un nodo hoja (no tiene hijos)
        if isempty(hgmm.topology.children[node])
            gmm = hgmm.gmms[node]
            for k in 1:gmm.K
                means_dict[(node, k)] = gmm.components[k].mu
            end
        end
    end
    
    return means_dict
end

# ===================================================================
# Funciones Utilitarias para Segmentación Directa
# ===================================================================

"""
    segment_data(gmm::GMM, X::Matrix{Float64}) -> Matrix{Float64}

Reemplaza cada punto en `X` con la media del componente al que pertenece.
Retorna una nueva matriz de datos segmentada.
"""
function segment_data(gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    means = get_means(gmm)
    assignments = predict(gmm, X)
    
    X_seg = similar(X)
    for n in 1:N
        X_seg[n, :] = means[assignments[n], :]
    end
    
    return X_seg
end

"""
    segment_data(hgmm::HGMM, X::Matrix{Float64}) -> Matrix{Float64}

Reemplaza cada punto en `X` con la media del componente hoja al que pertenece
según la estructura jerárquica del modelo.
"""
function segment_data(hgmm::HGMM, X::Matrix{Float64})
    N, D = size(X)
    means_dict = get_means(hgmm)
    leaf_assignments, comp_assignments = predict(hgmm, X)
    
    X_seg = similar(X)
    for n in 1:N
        # Usamos la tupla (nodo, componente) que predice el modelo para buscar el color
        nodo = leaf_assignments[n]
        comp = comp_assignments[n]
        X_seg[n, :] = means_dict[(nodo, comp)]
    end
    
    return X_seg
end
"""
    path_probability(hgmm::HGMM, X::Matrix{Float64}, leaf_node::Int, leaf_comp::Int) -> Vector{Float64}

Calcula la probabilidad (posterior) de que cada dato en `X` provenga de un camino 
específico del árbol. Un camino se define mediante su `leaf_node` (nodo final) 
y su `leaf_comp` (componente gaussiano final dentro de ese nodo).

Retorna un vector de tamaño N con las probabilidades P(camino | x) ∈ [0, 1].
"""
function path_probability(hgmm::HGMM, X::Matrix{Float64}, leaf_node::Int, leaf_comp::Int)
    N, D = size(X)
    
    # 1. Validar que el nodo sea una hoja y el componente exista
    if !isempty(hgmm.topology.children[leaf_node])
        error("El nodo $leaf_node no es un nodo hoja.")
    end
    if leaf_comp < 1 || leaf_comp > hgmm.gmms[leaf_node].K
        error("El componente $leaf_comp no es válido para el nodo hoja $leaf_node.")
    end
    
    # 2. Rastrear el camino desde la hoja hasta la raíz para sumar pesos de decisión
    path_log_weight = 0.0
    current_node = leaf_node
    
    while hgmm.topology.parents[current_node] > 0
        parent_node = hgmm.topology.parents[current_node]
        # Buscar qué índice de hijo es el nodo actual respecto a su padre
        child_idx = findfirst(==(current_node), hgmm.topology.children[parent_node])
        # Sumar el log-peso de la decisión del padre para bajar a este hijo
        path_log_weight += hgmm.gmms[parent_node].log_weights[child_idx]
        current_node = parent_node
    end
    
    # Sumar el log-peso del componente final dentro del propio nodo hoja
    path_log_weight += hgmm.gmms[leaf_node].log_weights[leaf_comp]
    
    # Extraer el gaussiano final de este camino
    final_gaussian = hgmm.gmms[leaf_node].components[leaf_comp]
    
    # 3. Preparar todos los caminos posibles para el denominador P(x)
    global_components = Gaussian[]
    global_log_weights = Float64[]
    
    function extract_leaves(node::Int, acc_log_weight::Float64)
        gmm = hgmm.gmms[node]
        if isempty(hgmm.topology.children[node]) # Es nodo hoja
            for k in 1:gmm.K
                push!(global_components, gmm.components[k])
                push!(global_log_weights, acc_log_weight + gmm.log_weights[k])
            end
        else
            for (k, child) in enumerate(hgmm.topology.children[node])
                extract_leaves(child, acc_log_weight + gmm.log_weights[k])
            end
        end
    end
    
    root_node = hgmm.topology.levels[1][1]
    extract_leaves(root_node, 0.0)
    
    # 4. Calcular la probabilidad para cada punto en la matriz X
    probabilities = Vector{Float64}(undef, N)
    
    for n in 1:N
        x = @view X[n, :]
        
        # A) Log-probabilidad del camino específico: log P(x, camino)
        log_joint_path = path_log_weight + logpdf(final_gaussian, x)
        
        # B) Log-probabilidad marginal (toda la evidencia): log P(x)
        log_px = -Inf
        for i in 1:length(global_components)
            lp = global_log_weights[i] + logpdf(global_components[i], x)
            if log_px == -Inf
                log_px = lp
            else
                # Log-sum-exp trick para estabilidad numérica
                log_px = lp > log_px ? lp + log1p(exp(log_px - lp)) : log_px + log1p(exp(lp - log_px))
            end
        end
        
        # C) Probabilidad posterior: P(camino | x) = exp( log P(x, camino) - log P(x) )
        probabilities[n] = exp(log_joint_path - log_px)
    end
    
    return probabilities
end
export get_means, segment_data, path_probability
end

#img_mu_GMM=[RGB(p1[i,1],p1[i,2],p1[i,3]) for (i,idx) in enumerate(img)]
