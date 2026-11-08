



using LinearAlgebra
using Random
using Statistics
using Printf



struct Gaussian
    mu::Vector{Float64}
    sigma::Matrix{Float64}

    precision::Matrix{Float64}    # Σ⁻¹ precomputada
    logdet::Float64               # log(det(Σ)) precomputado
    norm_const::Float64           # -d/2*log(2π) - 0.5*logdet
    dim::Int
end
"""
    Gaussian(mu, sigma; reg=1e-6)

Constructor que precomputa precisión, log-det y constante de normalización.
Añade regularización `reg*I` a la covarianza antes de factorizar.
"""
function Gaussian(mu::AbstractVector, sigma::AbstractMatrix; reg::Float64=1e-6)
    d = length(mu)
    @assert size(sigma) == (d, d) "Covarianza debe ser $(d)×$(d)"
    
    # Regularizar y factorizar con Cholesky (más estable que pinv)
    sigma_reg = sigma + reg * I
    F = cholesky(Symmetric(sigma_reg); check=false)
    if F.info != 0
        # Fallback: añadir más regularización
        sigma_reg = sigma + 1e-3 * I
        F = cholesky(Symmetric(sigma_reg); check=false)
        if F.info != 0
            error("Cholesky failed even after regularization")
        end
    end
    
    precision = F.U' \ (F.U \ I)  # Σ⁻¹ = U⁻ᵀ U⁻¹
    logdet = 2.0 * sum(log.(diag(F.U)))  # log(det(Σ)) = 2*sum(log(diag(U)))
    norm_const = -0.5 * d * log(2π) - 0.5 * logdet
    
    return Gaussian(mu, sigma_reg, precision, logdet, norm_const, d)
end

# Unidimensional
function Gaussian(mu::Real, sigma::Real; reg::Float64=1e-6)
    sigma_reg = sigma + reg
    precision = 1.0 / sigma_reg
    logdet = log(sigma_reg)
    norm_const = -0.5 * log(2π) - 0.5 * logdet
    return Gaussian([mu], [sigma_reg], [precision], logdet, norm_const, 1)
end

# ---------------------------------------------------------------------------
# Computo de Log-densidad (estable numéricamente)
# ---------------------------------------------------------------------------


@inline function logpdf(g::Gaussian, x::AbstractVector)
    diff = x .- g.mu
    # Mahalanobis distance: diff' * precision * diff
    mahal = dot(diff, g.precision * diff)
    return g.norm_const - 0.5 * mahal
end

@inline function logpdf(g::Gaussian, x::Real)
    diff = x - g.mu[1]
    return g.norm_const - 0.5 * g.precision[1] * diff * diff
end


# ---------------------------------------------------------------------------
# Mixture Model con estructura de Arbol para tener un mismo tipo entre GMM y HGMM.
# ---------------------------------------------------------------------------
# components, gausianas
# weights, pi
# K, numero de componentes
struct GMM
    components::Vector{Gaussian}
    weights::Vector{Float64}      # π_k (sum = 1)
    log_weights::Vector{Float64}  # log(π_k) precomputado
    K::Int
end


function GMM(components::Vector{Gaussian}, weights::AbstractVector)
    @assert length(components) == length(weights)
    logw = log.(weights)
    return GMM(components, weights, logw, length(components))
end
"""

Estructura de Arbol por Array para HGMM
uso:
```julia
Nodo 1 es raíz (0), 2 y 3 son hijos de 1, 4 y 5 son hijos de 2:
parents = [0, 1, 1, 2, 2]
```
"""
struct TreeTopology
    parents::Vector{Int}          # parents[i] nos da el id del padre del nodo i
    children::Vector{Vector{Int}} # children[i] lista de ids de los hijos del nodo i
    levels::Vector{Vector{Int}}   # Nodos agrupados por profundidad (para traversal por niveles)
end

function TreeTopology(parents::Vector{Int})
    n_nodes = length(parents)
    children = [Int[] for _ in 1:n_nodes]
    
    for (child_id, parent_id) in enumerate(parents)
        if parent_id > 0
            push!(children[parent_id], child_id)
        end
    end
    
    # Calcular niveles mediante BFS
    levels = Vector{Int}[]
    current_level = [findfirst(==(0), parents)] # Raíz
    
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


function e_step!(log_gamma::Matrix{Float64}, gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    log_like = 0.0
    
    @inbounds for n in 1:N
        x = @view X[n, :]
        # Log-numeradores: log πₖ + log N(xₙ|μₖ,Σₖ)
        log_num = Vector{Float64}(undef, K)
        for k in 1:K
            log_num[k] = gmm.log_weights[k] + logpdf(gmm.components[k], x)
        end
        
        # Log-sum-exp para normalizar y obtener log γₙₖ
        max_log = maximum(log_num)
        exp_sum = 0.0
        for k in 1:K
            exp_sum += exp(log_num[k] - max_log)
        end
        log_sum = max_log + log(exp_sum)
        
        for k in 1:K
            log_gamma[n, k] = log_num[k] - log_sum  # log γₙₖ
        end
        
        log_like += log_sum
    end
    
    return log_like
end

# Versión paralela (opcional, requiere Threads)
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

# ---------------------------------------------------------------------------
# M-step: actualiza μₖ, Σₖ, πₖ desde log-gamma
# ---------------------------------------------------------------------------

function m_step(gmm::GMM, X::Matrix{Float64}, log_gamma::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    
    new_components = Vector{Gaussian}(undef, K)
    new_weights = Vector{Float64}(undef, K)
    
    for k in 1:K
        # γₙₖ = exp(log_gamma[n,k])
        gamma_k = exp.(@view log_gamma[:, k])
        N_k = sum(gamma_k)
        
        # μₖ = (1/Nₖ) Σₙ γₙₖ xₙ
        mu_k = zeros(D)
        @inbounds for n in 1:N
            w = gamma_k[n]
            if w > 0
                mu_k .+= w * @view X[n, :]
            end
        end
        mu_k ./= N_k
        
        # Σₖ = (1/Nₖ) Σₙ γₙₖ (xₙ - μₖ)(xₙ - μₖ)' + reg*I
        sigma_k = zeros(D, D)
        @inbounds for n in 1:N
            w = gamma_k[n]
            if w > 0
                diff = @view(X[n, :]) .- mu_k
                sigma_k .+= w * (diff * diff')
            end
        end
        sigma_k ./= N_k
        
        # Regularización robusta: floor absoluto en autovalores + escala relativa
        # Evita colapso de componentes (problema conocido en GMM-EM)
        min_eig = 1e-3  # autovalor mínimo absoluto
        reg = max(min_eig, 1e-3 * tr(sigma_k) / D)
        sigma_k += reg * I
        
        # Peso mínimo para evitar componentes vacíos
        min_weight = 1e-3 / K
        new_weights[k] = max(N_k / N, min_weight)
        # Renormalizar pesos después del bucle
        
        new_components[k] = Gaussian(mu_k, sigma_k)
    end
    
    # Renormalizar pesos (debido a floor mínimo)
    new_weights ./= sum(new_weights)
    
    # Retornar nuevo GMM (inmutable)
    return GMM(new_components, new_weights)
end


function kmeans_plusplus_init(X::Matrix{Float64}, K::Int; rng::AbstractRNG=Random.default_rng())
    N, D = size(X)
    centers = Matrix{Float64}(undef, K, D)
    
    # 1er centro: punto aleatorio
    centers[1, :] = X[rand(rng, 1:N), :]
    
    # Distancias mínimas a centros existentes
    dist2 = fill(Inf, N)
    
    for k in 2:K
        # Actualizar dist² al centro más cercano
        @inbounds for n in 1:N
            d2 = sum(abs2, X[n, :] - centers[k-1, :])
            if d2 < dist2[n]
                dist2[n] = d2
            end
        end
        
        # Muestrear siguiente centro con prob ∝ dist²
        probs = dist2 ./ sum(dist2)
        idx = findfirst(cumsum(probs) .>= rand(rng))
        centers[k, :] = X[idx, :]
    end
    
    # Covarianza inicial: covarianza global de los datos
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
    
    # Inicialización K-means++
    gmm = kmeans_plusplus_init(X, K; rng=rng)
    
    log_gamma = Matrix{Float64}(undef, N, K)
    log_likelihoods = Float64[]
    
    log_like_prev = -Inf
    
    for iter in 1:max_iter
        # E-step
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
        
        # Convergencia
        if iter > 1 && abs(log_like - log_like_prev) < tol * abs(log_like_prev)
            verbose && println("  → Convergido (tol = $tol)")
            break
        end
        
        log_like_prev = log_like
        
        # M-step
        gmm = m_step(gmm, X, log_gamma)
    end
    
    return gmm, log_likelihoods
end


function fit_gmm(
    X::Matrix{Float64},
    K::TreeTopology;
    leafs::Union{Int,Vector{Int}}=2
    max_iter::Int=100,
    tol::Float64=1e-4,
    verbose::Bool=true,
    threaded::Bool=false,
    seed::Int=42
)
    rng = MersenneTwister(seed)
    N, D = size(X)
    
    # Inicialización K-means++
	#
#Recorrer estructura de arbol
	#INICIALIZACION
tam=length(K.children)
# Checar los componentes de cada hoja. Se puede decidir diferentes componentes o el mismo numero para todos al final
punt=length(K.levels[end])
	if leafs <: Int
	puntas=[leafs for _ in 1:punt]
	else 
	if length(leafs) == punt
		puntas=leafs
	else
	error("No tiene el mismo numero de componentes que hojas")
	end
	end
# Todos los GMM donde cada index(posicion) es equivalente al nodo al que pertenecen
gmms=Vector{GMM}(undef,tam)
## recorrido K_means para estructura de arbol

for nivel in estructura.levels[1:end-1]
	for i in nivel
		gmms[i] = kmeans_plusplus_init(X, length(K.children[i]); rng=rng)
end
## reocrrido K_means para las hojas 
for (idx,i) in enumerate(tam:punt)
	gmms[i] = kmeans_plusplus_init(X, puntas[idx]; rng=rng)
end



    log_gamma = Matrix{Float64}(undef, N, K)
    log_likelihoods = Float64[]
    
    log_like_prev = -Inf
    
    for iter in 1:max_iter
        # E-step
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
        
        # Convergencia
        if iter > 1 && abs(log_like - log_like_prev) < tol * abs(log_like_prev)
            verbose && println("  → Convergido (tol = $tol)")
            break
        end
        
        log_like_prev = log_like
        
        # M-step
        gmm = m_step(gmm, X, log_gamma)
    end
    
    return gmm, log_likelihoods
end

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
    predict_proba(gmm::GMM, X::Matrix{Float64}) -> Matrix{Float64}

Retorna la matriz de probabilidades posteriores (responsabilidades) γₙₖ = p(zₙ=k|xₙ).
Cada fila n suma a 1. Dimensiones: (N_puntos × K_componentes).
"""
function predict_proba(gmm::GMM, X::Matrix{Float64})
    N, D = size(X)
    K = gmm.K
    probs = Matrix{Float64}(undef, N, K)
    
    for n in 1:N
        x = @view X[n, :]
        # Log-numeradores: log πₖ + log N(xₙ|μₖ,Σₖ)
        log_num = Vector{Float64}(undef, K)
        for k in 1:K
            log_num[k] = gmm.log_weights[k] + logpdf(gmm.components[k], x)
        end
        
        # Log-sum-exp para normalizar
        max_log = maximum(log_num)
        exp_sum = 0.0
        for k in 1:K
            exp_sum += exp(log_num[k] - max_log)
        end
        log_sum = max_log + log(exp_sum)
        
        for k in 1:K
            probs[n, k] = exp(log_num[k] - log_sum)  # γₙₖ
        end
    end
    
    return probs
end

"""
    predict_with_confidence(gmm::GMM, X::Matrix{Float64}) -> (assignments, confidences)

Retorna tupla: (vector de clases asignadas, vector de probabilidad/confianza de la clase ganadora).
La confianza es el porcentaje (0-100) de la clase más probable.
"""
function predict_with_confidence(gmm::GMM, X::Matrix{Float64})
    probs = predict_proba(gmm, X)
    N, K = size(probs)
    assignments = Vector{Int}(undef, N)
    confidences = Vector{Float64}(undef, N)
    
    for n in 1:N
        best_k = argmax(probs[n, :])
        assignments[n] = best_k
        confidences[n] = probs[n, best_k] * 100.0  # porcentaje
    end
    
    return assignments, confidences
end

"""Log-verosimilitud de datos bajo el modelo"""
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
                # log-sum-exp
                if lp > log_sum
                    log_sum = lp + log1p(exp(log_sum - lp))
                else
                    log_sum = log_sum + log1p(exp(lp - log_sum))
                end
            end
        end
        ll += log_sum
    end
    return ll
end
