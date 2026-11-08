# ============================================================================
# GLCM_Optimizado.jl - Matriz de Co-ocurrencia Optimizada en Julia
# ============================================================================
# 
# Implementación optimizada de GLCM usando:
# 1. Struct one_hot con UInt8 para 256 combinaciones (16×16 niveles de gris)
# 2. Cuantización a 16 niveles de gris
# 3. Desplazamiento direccional (theta, rho) con manejo de bordes
# 4. Codificación one-hot mediante multiplicación elemento a elemento
# 5. ImageFiltering.jl + Kernel.box para conteo eficiente en ventana nxn
# 6. Aplicación directa de propiedades de Haralick sin almacenar GLCM completa
# 7. Versión optimizada con histogramas integrales (pasada única)
# ============================================================================

module GLCM_Optimizado

using Images, ImageFiltering, ImageCore, ImageFiltering.KernelFactors
using Statistics, LinearAlgebra, TestImages, Base.Threads

# ============================================================================
# 1. STRUCT ONE_HOT - Codificación one-hot en UInt8 (256 valores = 16×16)
# ============================================================================

"""
    one_hot

Struct que encapsula un UInt8 para representar una de las 256 combinaciones
posibles de pares de niveles de gris (16 × 16 = 256).

El valor se codifica como: `value = i * 16 + j` donde i, j ∈ 0:15

Ventajas de memoria:
- 1 byte por píxel vs 256 bytes (vector completo GLCM)
- Operaciones vectorizadas eficientes en Julia
"""
struct one_hot
    value::UInt8
end

# Constructores convenientes
one_hot(i::Integer, j::Integer) = one_hot(UInt8(i * 16 + j))
# one_hot(v::UInt8) usa el constructor por defecto del struct
Base.UInt8(oh::one_hot) = oh.value
Base.Int(oh::one_hot) = Int(oh.value)
Base.convert(::Type{UInt8}, oh::one_hot) = oh.value
Base.convert(::Type{Int}, oh::one_hot) = Int(oh.value)
Base.show(io::IO, oh::one_hot) = print(io, "one_hot($(oh.value))")
Base.:(==)(a::one_hot, b::one_hot) = a.value == b.value
Base.hash(oh::one_hot, h::UInt) = hash(oh.value, h)

# Operaciones necesarias para ImageFiltering.imfilter
# imfilter requiere: zero(T), +(T, T), *(T, Number)
Base.zero(::Type{one_hot}) = one_hot(0x00)
Base.:+(a::one_hot, b::one_hot) = one_hot(a.value + b.value)  # Suma módulo 256
Base.:*(a::one_hot, k::Number) = one_hot(a.value * UInt8(k))  # Escalado
Base.:*(k::Number, a::one_hot) = a * k

# ============================================================================
# 2. CUANTIZACIÓN A 16 NIVELES DE GRIS
# ============================================================================

"""
    quantize_16levels(img::Matrix{<:Real}) -> Matrix{UInt8}

Cuantiza una imagen en escala de grises a 16 niveles (0-15).
Maneja cualquier rango de valores de entrada normalizando a [0, 15].
"""
function quantize_16levels(img::Matrix{<:Real})::Matrix{UInt8}
    h, w = size(img)
    img_min, img_max = minimum(img), maximum(img)
    
    if img_max > img_min
        # Normalizar a [0, 1] y escalar a [0, 15]
        q = round.(UInt8, (img .- img_min) ./ (img_max - img_min) .* 15)
    else
        q = zeros(UInt8, h, w)
    end
    
    clamp!(q, 0x00, 0x0f)
    return q
end

# ============================================================================
# 3. DESPLAZAMIENTO DIRECCIONAL (THETA, RHO) CON MANEJO DE BORDES
# ============================================================================

"""
    directional_shift(img::Matrix{UInt8}, theta::Real, rho::Int) -> Matrix{UInt8}

Desplaza la imagen en dirección `theta` (radianes) y distancia `rho` (píxeles).
Maneja bordes rellenando con 0 (nivel de gris 0) y mantiene el tamaño original.

Args:
    img: Imagen cuantizada (UInt8 0-15)
    theta: Ángulo en radianes (0=horizontal, π/4=diagonal, π/2=vertical, 3π/4=anti-diagonal)
    rho: Distancia en píxeles (típicamente 1)

Returns:
    Imagen desplazada del mismo tamaño, con ceros en bordes donde no hay vecino
"""
function directional_shift(img::Matrix{UInt8}, theta::Real, rho::Int)::Matrix{UInt8}
    h, w = size(img)
    dx = round(Int, rho * cos(theta))
    dy = round(Int, rho * sin(theta))
    
    shifted = zeros(UInt8, h, w)
    
    # Rangos válidos de origen para que destino caiga dentro de la imagen
    y_src_start = max(1, 1 - dy)
    y_src_end = min(h, h - dy)
    x_src_start = max(1, 1 - dx)
    x_src_end = min(w, w - dx)
    
    y_dst_start = y_src_start + dy
    y_dst_end = y_src_end + dy
    x_dst_start = x_src_start + dx
    x_dst_end = x_src_end + dx
    
    if y_src_start <= y_src_end && x_src_start <= x_src_end
        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = 
            img[y_src_start:y_src_end, x_src_start:x_src_end]
    end
    
    return shifted
end

# ============================================================================
# 4. CODIFICACIÓN ONE-HOT MEDIANTE MULTIPLICACIÓN ELEMENTO A ELEMENTO
# ============================================================================

"""
    encode_one_hot(img::Matrix{UInt8}, shifted::Matrix{UInt8}) -> Matrix{one_hot}

Codifica pares de píxeles (original, desplazado) como one_hot.
El valor se calcula como: `value = img * 16 + shifted` (0-255).

Esta es la clave de la optimización: en lugar de almacenar una matriz GLCM
de 256×256 por píxel, codificamos el par como un solo UInt8.
"""
function encode_one_hot(img::Matrix{UInt8}, shifted::Matrix{UInt8})::Matrix{one_hot}
    h, w = size(img)
    oh_matrix = Matrix{one_hot}(undef, h, w)
    
    @inbounds @threads for y in 1:h
        for x in 1:w
            # i = img[y,x] ∈ 0:15, j = shifted[y,x] ∈ 0:15
            # value = i * 16 + j ∈ 0:255
            oh_matrix[y, x] = one_hot(img[y, x] * 0x10 + shifted[y, x])
        end
    end
    
    return oh_matrix
end

# ============================================================================
# 5. FILTRADO CON KERNEL BOX - VERSIÓN CON IMAGENES BINARIAS (256 PASADAS)
# ============================================================================

"""
    box_filter_counts_imfilter(oh_matrix::Matrix{one_hot}, window_size::Int) -> Matrix{Vector{Int}}

Aplica filtro box (ventana cuadrada) usando ImageFiltering.imfilter para contar
ocurrencias de cada uno de los 256 valores one_hot en cada ventana.

⚠ ADVERTENCIA: Esta versión hace 256 llamadas a imfilter - es LENTA.
Se incluye para demostrar el enfoque solicitado con ImageFiltering.jl.
Para producción usar `box_filter_counts_optimized`.

Returns:
    Matriz H×W donde cada elemento es un Vector{Int} de 256 conteos.
"""
function box_filter_counts_imfilter(oh_matrix::Matrix{one_hot}, window_size::Int)::Matrix{Vector{Int}}
    h, w = size(oh_matrix)
    
    # Kernel box: todos unos, normalizado para que imfilter sume
    kernel = Kernel.box((window_size, window_size))
    
    counts_matrix = Matrix{Vector{Int}}(undef, h, w)
    
    println("  Aplicando filtro box ($window_size×$window_size) con imfilter (256 pasadas)...")
    
    # Buffers reutilizables
    mask = Matrix{Float64}(undef, h, w)
    filtered = Matrix{Float64}(undef, h, w)
    
    # Inicializar vectores de conteo
    @inbounds for y in 1:h, x in 1:w
        counts_matrix[y, x] = zeros(Int, 256)
    end
    
    for v in 0x00:0xff
        # Crear máscara binaria: 1.0 donde oh_matrix == v, 0.0 en otro caso
        @inbounds @threads for y in 1:h
            for x in 1:w
                mask[y, x] = (oh_matrix[y, x].value == v) ? 1.0 : 0.0
            end
        end
        
        # Aplicar filtro box - suma los unos en la ventana
        # Fill(0.0) rellena bordes con ceros
        imfilter!(filtered, mask, kernel, Fill(0.0))
        
        # Acumular conteos
        @inbounds @threads for y in 1:h
            for x in 1:w
                counts_matrix[y, x][v + 1] = round(Int, filtered[y, x])
            end
        end
        
        if v % 0x40 == 0
            println("    Progreso: $(v+1)/256")
        end
    end
    
    return counts_matrix
end

# ============================================================================
# 6. FILTRADO OPTIMIZADO - HISTOGRAMAS INTEGRALES (PASADA ÚNICA)
# ============================================================================

"""
    box_filter_counts_optimized(oh_matrix::Matrix{one_hot}, window_size::Int) -> Matrix{Vector{Int}}

Versión optimizada: una sola pasada usando histogramas integrales (integral histograms).
Mucho más rápido que 256 llamadas a imfilter.

Algoritmo:
1. Para cada valor v ∈ 0..255, construye imagen integral del indicador (v == oh_matrix)
2. Para cada píxel, consulta el rectángulo en la imagen integral en O(1)
3. Total: 256 imágenes integrales + consultas O(1) por píxel

Complejidad: O(256 × H × W) tiempo, O(256 × H × W) memoria
vs imfilter: O(256 × H × W × window_size²) tiempo
"""
function box_filter_counts_optimized(oh_matrix::Matrix{one_hot}, window_size::Int)::Matrix{Vector{Int}}
    h, w = size(oh_matrix)
    half_w = window_size ÷ 2
    
    # Histograma integral: para cada valor v, integral_image[v][y,x] = count of v in rectangle (1,1) to (y,x)
    # Usamos indexación 1-based con fila/columna 0 rellena de ceros
    integral_hists = [zeros(Int, h+1, w+1) for _ in 1:256]
    
    println("  Construyendo histogramas integrales (pasada única)...")
    
    # Construir histogramas integrales - UNA sola pasada por la imagen
    # Fórmula: I[y+1, x+1] = I[y+1, x] + I[y, x+1] - I[y, x] + (img[y,x] == v ? 1 : 0)
    @inbounds for y in 1:h
        # Optimización: calcular fila por fila, reutilizando valores previos
        for x in 1:w
            v = oh_matrix[y, x].value + 1  # 1-based index
            for val in 1:256
                integral_hists[val][y+1, x+1] = integral_hists[val][y+1, x] + 
                                                 integral_hists[val][y, x+1] - 
                                                 integral_hists[val][y, x] + 
                                                 (val == v ? 1 : 0)
            end
        end
    end
    
    # Extraer conteos para cada ventana usando histogramas integrales
    # Ventana centrada en (y,x): desde (y-half_w, x-half_w) hasta (y+half_w, x+half_w)
    # En coordenadas de imagen integral (1-based con padding):
    # y1 = y - half_w, x1 = x - half_w, y2 = y + half_w, x2 = x + half_w
    # count = I[y2+1, x2+1] - I[y1, x2+1] - I[y2+1, x1] + I[y1, x1]
    counts_matrix = Matrix{Vector{Int}}(undef, h, w)
    
    println("  Extrayendo conteos de ventanas...")
    
    @inbounds @threads for y in 1:h
        for x in 1:w
            y1 = max(1, y - half_w)
            y2 = min(h, y + half_w)
            x1 = max(1, x - half_w)
            x2 = min(w, x + half_w)
            
            # Convertir a índices de imagen integral (offset +1)
            iy1, iy2 = y1, y2 + 1
            ix1, ix2 = x1, x2 + 1
            
            counts = zeros(Int, 256)
            for val in 1:256
                I = integral_hists[val]
                counts[val] = I[iy2, ix2] - I[iy1, ix2] - I[iy2, ix1] + I[iy1, ix1]
            end
            counts_matrix[y, x] = counts
        end
    end
    
    return counts_matrix
end

# ============================================================================
# 7. PROPIEDADES DE HARALICK - APLICACIÓN DIRECTA SOBRE CONTEOS
# ============================================================================

"""
    AbstractHaralickProperty

Tipo abstracto para propiedades de Haralick. Permite pasar funciones arbitrarias.
"""
abstract type AbstractHaralickProperty end

"""
    HaralickProperty

Wrapper para funciones de propiedades de Haralick.
"""
struct HaralickProperty{F} <: AbstractHaralickProperty
    name::String
    func::F
end

HaralickProperty(name::String, func::F) where F = HaralickProperty{F}(name, func)
(name::HaralickProperty)(counts::Vector{Int}) = name.func(counts)

# Propiedades de Haralick predefinidas

"""
    haralick_entropy(counts) -> Float64

Entropía: -Σ p(i,j) log p(i,j) — mide aleatoriedad/complejidad
"""
function haralick_entropy(counts::Vector{Int})::Float64
    total = sum(counts)
    total == 0 && return 0.0
    entropy = 0.0
    @inbounds for c in counts
        if c > 0
            p = c / total
            entropy -= p * log(p)
        end
    end
    return entropy
end

"""
    haralick_contrast(counts) -> Float64

Contraste: Σ |i-j|² p(i,j) — variación local de intensidad
"""
function haralick_contrast(counts::Vector{Int})::Float64
    total = sum(counts)
    total == 0 && return 0.0
    contrast = 0.0
    @inbounds for idx in 1:256
        c = counts[idx]
        if c > 0
            i = (idx - 1) ÷ 16  # 0-15
            j = (idx - 1) % 16  # 0-15
            p = c / total
            contrast += (i - j)^2 * p
        end
    end
    return contrast
end

"""
    haralick_idm(counts) -> Float64

IDM (Inverse Difference Moment) / Homogeneidad: Σ p(i,j) / (1 + |i-j|²)
Mide homogeneidad — alto en regiones uniformes
"""
function haralick_idm(counts::Vector{Int})::Float64
    total = sum(counts)
    total == 0 && return 0.0
    idm = 0.0
    @inbounds for idx in 1:256
        c = counts[idx]
        if c > 0
            i = (idx - 1) ÷ 16
            j = (idx - 1) % 16
            p = c / total
            idm += p / (1.0 + (i - j)^2)
        end
    end
    return idm
end

"""
    haralick_asm(counts) -> Float64

ASM (Angular Second Moment) / Energía: Σ p(i,j)² — mide uniformidad
"""
function haralick_asm(counts::Vector{Int})::Float64
    total = sum(counts)
    total == 0 && return 0.0
    asm = 0.0
    @inbounds for c in counts
        if c > 0
            p = c / total
            asm += p * p
        end
    end
    return asm
end

"""
    haralick_correlation(counts) -> Float64

Correlación: medida de dependencia lineal entre píxeles vecinos
"""
function haralick_correlation(counts::Vector{Int})::Float64
    total = sum(counts)
    total == 0 && return 0.0
    
    # Probabilidades marginales
    px = zeros(Float64, 16)
    py = zeros(Float64, 16)
    
    @inbounds for idx in 1:256
        c = counts[idx]
        if c > 0
            i = (idx - 1) ÷ 16
            j = (idx - 1) % 16
            p = c / total
            px[i+1] += p
            py[j+1] += p
        end
    end
    
    # Medias
    μx = sum(i * px[i] for i in 1:16)
    μy = sum(j * py[j] for j in 1:16)
    
    # Varianzas
    σx² = sum((i - μx)^2 * px[i] for i in 1:16)
    σy² = sum((j - μy)^2 * py[j] for j in 1:16)
    
    σx = sqrt(σx²)
    σy = sqrt(σy²)
    
    σx * σy == 0 && return 0.0
    
    # Covarianza
    cov = 0.0
    @inbounds for idx in 1:256
        c = counts[idx]
        if c > 0
            i = (idx - 1) ÷ 16
            j = (idx - 1) % 16
            p = c / total
            cov += (i - μx) * (j - μy) * p
        end
    end
    
    return cov / (σx * σy)
end

# Propiedades predefinidas como instancias
const HARALICK_ENTROPY    = HaralickProperty("entropy", haralick_entropy)
const HARALICK_CONTRAST   = HaralickProperty("contrast", haralick_contrast)
const HARALICK_IDM        = HaralickProperty("idm", haralick_idm)
const HARALICK_ASM        = HaralickProperty("asm", haralick_asm)
const HARALICK_CORRELATION = HaralickProperty("correlation", haralick_correlation)

# ============================================================================
# 8. FUNCIÓN PRINCIPAL: GLCM OPTIMIZADO SINGLE-ÁNGULO
# ============================================================================

"""
    glcm_optimized(img::Matrix{<:Real}; 
                   theta::Real=0.0, 
                   rho::Int=1, 
                   window_size::Int=7,
                   properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM],
                   use_optimized::Bool=true) -> NamedTuple

Función principal para GLCM optimizado en un solo ángulo.

Pipeline:
1. Cuantizar imagen a 16 niveles de gris
2. Desplazar imagen según (theta, rho)
3. Codificar pares como one_hot (multiplicación elemento a elemento)
4. Contar ocurrencias en ventana nxn (imfilter o histogramas integrales)
5. Aplicar propiedades de Haralick directamente a los conteos
6. Retornar matrices de propiedades (NO la GLCM completa)

Args:
    img: Imagen en escala de grises (cualquier rango)
    theta: Ángulo en radianes (0, π/4, π/2, 3π/4)
    rho: Distancia en píxeles (default 1)
    window_size: Tamaño de ventana cuadrada (impar, default 7)
    properties: Vector de propiedades de Haralick a calcular
    use_optimized: true=histogramas integrales (rápido), false=imfilter (lento, como se pidió)

Returns:
    NamedTuple con matrices de propiedades (ej: (entropy=Matrix, contrast=Matrix, ...))
"""
function glcm_optimized(img::Matrix{<:Real}; 
                        theta::Real=0.0, 
                        rho::Int=1, 
                        window_size::Int=7,
                        properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM],
                        use_optimized::Bool=true)::NamedTuple
    
    println("=== GLCM Optimizado (θ=$(rad2deg(theta))°, ρ=$rho, window=$window_size×$window_size) ===")
    
    # 1. Cuantizar a 16 niveles
    println("1. Cuantizando a 16 niveles de gris...")
    img_q = quantize_16levels(img)
    println("   Rango: $(minimum(img_q)) - $(maximum(img_q))")
    
    # 2. Desplazamiento direccional
    println("2. Aplicando desplazamiento direccional...")
    shifted = directional_shift(img_q, theta, rho)
    
    # 3. Codificación one-hot
    println("3. Codificando pares como one_hot (UInt8)...")
    oh_matrix = encode_one_hot(img_q, shifted)
    
    # 4. Conteo en ventana
    println("4. Contando ocurrencias en ventana $window_size×$window_size...")
    if use_optimized
        counts_matrix = box_filter_counts_optimized(oh_matrix, window_size)
    else
        counts_matrix = box_filter_counts_imfilter(oh_matrix, window_size)
    end
    
    # 5. Aplicar propiedades de Haralick directamente
    println("5. Calculando $(length(properties)) propiedades de Haralick...")
    h, w = size(img_q)
    
    # Pre-asignar matrices de salida
    result_dict = Dict{String, Matrix{Float64}}()
    for prop in properties
        result_dict[prop.name] = Matrix{Float64}(undef, h, w)
    end
    
    @inbounds @threads for y in 1:h
        for x in 1:w
            counts = counts_matrix[y, x]
            for prop in properties
                result_dict[prop.name][y, x] = prop(counts)
            end
        end
    end
    
    println("   Completado.")
    
    # Convertir a NamedTuple - las keys ya son Strings, convertimos a Symbols
    keys_vec = collect(keys(result_dict))
    keys_sym = Tuple(Symbol.(keys_vec))
    vals = Tuple(result_dict[k] for k in keys_vec)
    return NamedTuple{keys_sym}(vals)
end

# ============================================================================
# 9. GLCM MULTI-ÁNGULO (PROMEDIO DE PROPIEDADES) - OPTIMIZADO
# ============================================================================

"""
    glcm_multi_angle_optimized(img::Matrix{<:Real};
                               angles::Vector{<:Real}=[0, π/4, π/2, 3π/4],
                               rho::Int=1,
                               window_size::Int=7,
                               properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM],
                               use_optimized::Bool=true) -> NamedTuple

Calcula GLCM optimizado promediando propiedades sobre múltiples ángulos.

OPTIMIZACIONES:
- Cuantización a 16 niveles: **UNA SOLA VEZ** al inicio
- Para cada ángulo: solo desplazamiento + one_hot + histograma + propiedades
- No re-cuantiza ni re-calcula lo invariante al ángulo
"""
function glcm_multi_angle_optimized(img::Matrix{<:Real};
                                    angles::Vector{<:Real}=[0, π/4, π/2, 3π/4],
                                    rho::Int=1,
                                    window_size::Int=7,
                                    properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM],
                                    use_optimized::Bool=true)::NamedTuple
    
    println("=== GLCM Multi-Ángulo Optimizado ($(length(angles)) ángulos) ===")
    
    # 1. CUANTIZACIÓN UNA SOLA VEZ (invariante al ángulo)
    println("1. Cuantizando a 16 niveles de gris (una vez)...")
    img_q = quantize_16levels(img)
    println("   Rango: $(minimum(img_q)) - $(maximum(img_q))")
    
    h, w = size(img_q)
    
    # Acumuladores para cada propiedad
    accumulators = Dict{String, Matrix{Float64}}()
    for prop in properties
        accumulators[prop.name] = zeros(Float64, h, w)
    end
    
    # 2. PARA CADA ÁNGULO: solo shift + one_hot + histograma + props
    for (i, theta) in enumerate(angles)
        println("\n--- Ángulo $i/$(length(angles)): θ=$(rad2deg(theta))° ---")
        
        # 2a. Desplazamiento direccional (depende del ángulo)
        println("   Desplazamiento direccional...")
        shifted = directional_shift(img_q, theta, rho)
        
        # 2b. Codificación one-hot (depende del ángulo via shifted)
        println("   Codificando one_hot...")
        oh_matrix = encode_one_hot(img_q, shifted)
        
        # 2c. Histograma integral + conteo en ventana
        println("   Histograma integral + conteo ventana...")
        if use_optimized
            counts_matrix = box_filter_counts_optimized(oh_matrix, window_size)
        else
            counts_matrix = box_filter_counts_imfilter(oh_matrix, window_size)
        end
        
        # 2d. Aplicar propiedades directamente y acumular
        println("   Calculando y acumulando propiedades...")
        @inbounds @threads for y in 1:h
            for x in 1:w
                counts = counts_matrix[y, x]
                for prop in properties
                    accumulators[prop.name][y, x] += prop(counts)
                end
            end
        end
    end
    
    # 3. Promediar sobre ángulos
    println("\nPromediando sobre $(length(angles)) ángulos...")
    for prop in properties
        accumulators[prop.name] ./= length(angles)
    end
    
    keys_vec = collect(keys(accumulators))
    keys_sym = Tuple(Symbol.(keys_vec))
    vals = Tuple(accumulators[k] for k in keys_vec)
    return NamedTuple{keys_sym}(vals)
end

# ============================================================================
# 10. UTILIDADES DE NORMALIZACIÓN Y VISUALIZACIÓN
# ============================================================================

"""
    normalize01(mat::Matrix{<:Real}) -> Matrix{Float64}

Normaliza matriz a rango [0, 1] para visualización.
"""
function normalize01(mat::Matrix{<:Real})::Matrix{Float64}
    mn, mx = minimum(mat), maximum(mat)
    if mx > mn
        return Float64.(mat .- mn) ./ (mx - mn)
    else
        return zeros(Float64, size(mat))
    end
end

"""
    save_property_maps(result::NamedTuple, prefix::String="result")

Guarda cada matriz de propiedades como imagen PNG normalizada.
"""
function save_property_maps(result::NamedTuple, prefix::String="result")
    for (name, mat) in pairs(result)
        normed = normalize01(mat)
        img_rgb = colorview(RGB, normed, normed, normed)
        filename = "$(prefix)_$(name).png"
        save(filename, img_rgb)
        println("  Guardado: $filename (min=$(round(minimum(mat), digits=4)), max=$(round(maximum(mat), digits=4)))")
    end
end

# ============================================================================
# 11. GLCM HÍBRIDO - VENTANA DESLIZANTE DIRECTA + ONE_HOT (MÁS RÁPIDO)
# ============================================================================

"""
    glcm_hybrid_single(img::Matrix{<:Real};
                       theta::Real=0.0,
                       rho::Int=1,
                       window_size::Int=7,
                       properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM]) -> NamedTuple

GLCM híbrido: ventana deslizante directa (como GLCM_puro) + one_hot encoding + propiedades directas.

VENTAJAS vs integral histogram:
- Sin 256 imágenes integrales (ahorra 268 MB + tiempo construcción)
- Loop simple O(window²) por píxel, muy optimizable con @inbounds @simd
- Reutiliza un solo histograma de 256 bins por thread (sin allocations)
- Cuantización y propiedades como la versión optimizada

Para window=7: 49 iteraciones/píxel vs 256 accesos memoria integral histogram
"""
function glcm_hybrid_single(img::Matrix{<:Real};
                            theta::Real=0.0,
                            rho::Int=1,
                            window_size::Int=7,
                            properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM])::NamedTuple
    
    println("=== GLCM Híbrido (θ=$(rad2deg(theta))°, ρ=$rho, window=$window_size×$window_size) ===")
    
    # 1. Cuantizar a 16 niveles
    println("1. Cuantizando a 16 niveles de gris...")
    img_q = quantize_16levels(img)
    println("   Rango: $(minimum(img_q)) - $(maximum(img_q))")
    
    # 2. Desplazamiento direccional
    println("2. Aplicando desplazamiento direccional...")
    shifted = directional_shift(img_q, theta, rho)
    
    # 3. Codificación one-hot (value = i*16 + j, donde i,j ∈ 0:15)
    println("3. Codificando pares como one_hot (UInt8)...")
    oh_matrix = encode_one_hot(img_q, shifted)
    
    h, w = size(img_q)
    half_w = window_size ÷ 2
    
    # Pre-asignar matrices de salida
    result_dict = Dict{String, Matrix{Float64}}()
    for prop in properties
        result_dict[prop.name] = Matrix{Float64}(undef, h, w)
    end
    
    println("4. Ventana deslizante directa + propiedades ($window_size×$window_size)...")
    
    # Histograma compartido por thread (256 bins) - se reinicia por píxel
    # Usamos Base.Threads.threadid() para thread-local storage implícito
    
    @inbounds @threads for y in 1:h
        # Histograma local por thread (evita false sharing)
        local_hist = zeros(Int, 256)
        
        for x in 1:w
            # Limpiar histograma
            fill!(local_hist, 0)
            
            # Ventana: [y-half_w : y+half_w] × [x-half_w : x+half_w]
            y1 = max(1, y - half_w)
            y2 = min(h, y + half_w)
            x1 = max(1, x - half_w)
            x2 = min(w, x + half_w)
            
            # Iterar píxeles en ventana y acumular en histograma
            @inbounds @simd for yy in y1:y2
                for xx in x1:x2
                    v = oh_matrix[yy, xx].value + 1  # 1-based
                    local_hist[v] += 1
                end
            end
            
            # Calcular propiedades directamente sobre histograma
            for prop in properties
                result_dict[prop.name][y, x] = prop(local_hist)
            end
        end
    end
    
    println("   Completado.")
    
    keys_vec = collect(keys(result_dict))
    keys_sym = Tuple(Symbol.(keys_vec))
    vals = Tuple(result_dict[k] for k in keys_vec)
    return NamedTuple{keys_sym}(vals)
end


"""
    glcm_hybrid_multi(img::Matrix{<:Real};
                      angles::Vector{<:Real}=[0, π/4, π/2, 3π/4],
                      rho::Int=1,
                      window_size::Int=7,
                      properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM]) -> NamedTuple

GLCM híbrido multi-ángulo: cuantiza UNA vez, luego para cada ángulo hace shift+one_hot+ventana+props.

OPTIMIZACIONES:
- Cuantización: 1 vez (invariante al ángulo)
- Por ángulo: shift + one_hot + ventana directa + props
- Reutiliza histograma local por thread
- Acumula propiedades y promedia al final
"""
function glcm_hybrid_multi(img::Matrix{<:Real};
                           angles::Vector{<:Real}=[0, π/4, π/2, 3π/4],
                           rho::Int=1,
                           window_size::Int=7,
                           properties::Vector{<:AbstractHaralickProperty}=[HARALICK_ENTROPY, HARALICK_CONTRAST, HARALICK_IDM])::NamedTuple
    
    println("=== GLCM Híbrido Multi-Ángulo ($(length(angles)) ángulos) ===")
    
    # 1. CUANTIZACIÓN UNA SOLA VEZ
    println("1. Cuantizando a 16 niveles de gris (una vez)...")
    img_q = quantize_16levels(img)
    println("   Rango: $(minimum(img_q)) - $(maximum(img_q))")
    
    h, w = size(img_q)
    half_w = window_size ÷ 2
    
    # Acumuladores para cada propiedad
    accumulators = Dict{String, Matrix{Float64}}()
    for prop in properties
        accumulators[prop.name] = zeros(Float64, h, w)
    end
    
    # 2. PARA CADA ÁNGULO: shift + one_hot + ventana + props
    for (i, theta) in enumerate(angles)
        println("\n--- Ángulo $i/$(length(angles)): θ=$(rad2deg(theta))° ---")
        
        # 2a. Desplazamiento direccional
        println("   Desplazamiento direccional...")
        shifted = directional_shift(img_q, theta, rho)
        
        # 2b. Codificación one-hot
        println("   Codificando one_hot...")
        oh_matrix = encode_one_hot(img_q, shifted)
        
        # 2c. Ventana deslizante directa + propiedades
        println("   Ventana deslizante + propiedades...")
        
        @inbounds @threads for y in 1:h
            local_hist = zeros(Int, 256)
            
            for x in 1:w
                fill!(local_hist, 0)
                
                y1 = max(1, y - half_w)
                y2 = min(h, y + half_w)
                x1 = max(1, x - half_w)
                x2 = min(w, x + half_w)
                
                @inbounds @simd for yy in y1:y2
                    for xx in x1:x2
                        v = oh_matrix[yy, xx].value + 1
                        local_hist[v] += 1
                    end
                end
                
                for prop in properties
                    accumulators[prop.name][y, x] += prop(local_hist)
                end
            end
        end
    end
    
    # 3. Promediar
    println("\nPromediando sobre $(length(angles)) ángulos...")
    for prop in properties
        accumulators[prop.name] ./= length(angles)
    end
    
    keys_vec = collect(keys(accumulators))
    keys_sym = Tuple(Symbol.(keys_vec))
    vals = Tuple(accumulators[k] for k in keys_vec)
    return NamedTuple{keys_sym}(vals)
end

# ============================================================================
# EXPORTAR NUEVAS FUNCIONES
# ============================================================================
export glcm_hybrid_single, glcm_hybrid_multi

end # module

"""
val3=hcat([vec(propiedades3[i]) for i in 1:length(propiedades3)]...)
val5=hcat([vec(propiedades5[i]) for i in 1:length(propiedades3)]...)
val7=hcat([vec(propiedades7[i]) for i in 1:length(propiedades3)]...)
datos=hcat(val3,val5,val7)
color=[RGB(1,0,0),RGB(0,1,0),RGB(0,0,1),RGB(1,1,0),RGB(1,0,1),RGB(0,1,1),RGB(1,0.5,0.5)]
res=predict(modelo,datos)
imagen=[ color[res[idx]] for (idx,i) in enumerate(img)]
"""
funcs=[GLCM_Optimizado.HARALICK_ENTROPY,GLCM_Optimizado.HARALICK_ASM,GLCM_Optimizado.HARALICK_IDM,GLCM_Optimizado.HARALICK_CONTRAST]

#funcs=[GLCM_Optimizado.HARALICK_ENTROPY,GLCM_Optimizado.HARALICK_IDM]

function GLCM_diff(img;funcs=nothing)
if isnothing(funcs)
val3=glcm_hybrid_multi(img;window_size=3)
val7=glcm_hybrid_multi(img;window_size=31)
else
val3=glcm_hybrid_multi(img;window_size=3,properties=funcs)
val7=glcm_hybrid_multi(img;window_size=31,properties=funcs)
end
    datos=hcat(vec(val3[1]),vec(val7[1]))

for i in 2:length(val3)
        datos=hcat(datos,vec(val3[i]),vec(val7[i]))
end
for i in 1:length(val3)
        dif=val3[i] .- val7[1]
        datos=hcat(datos,vec(dif))
end
return datos
end 
function GLCM_vent(img;funcs=nothing)
if isnothing(funcs)
val3=glcm_hybrid_multi(img;window_size=3)
val5=glcm_hybrid_multi(img;window_size=21)
val7=glcm_hybrid_multi(img;window_size=41)
else
val3=glcm_hybrid_multi(img;window_size=3,properties=funcs)
val5=glcm_hybrid_multi(img;window_size=21,properties=funcs)
val7=glcm_hybrid_multi(img;window_size=41,properties=funcs)
end
   datos=hcat(vec(val3[1]),vec(val5[1]),vec(val7[1]))
    #datos=hcat(vec(val3[1]),vec(val7[1]))

for i in 2:length(val3)
       datos=hcat(datos,vec(val3[i]),vec(val5[i]),vec(val7[i]))
    #datos=hcat(datos,vec(val3[i]),vec(val7[i]))
end
return datos
end 

function GLCM_vent_dif1(img)
val3=glcm_hybrid_multi(img;window_size=3,properties=[GLCM_Optimizado.HARALICK_ENTROPY,GLCM_Optimizado.HARALICK_ASM]
)
# val5=glcm_hybrid_multi(img;window_size=21,properties=[GLCM_Optimizado.HARALICK_ENTROPY,GLCM_Optimizado.HARALICK_CORRELATION])
val7=glcm_hybrid_multi(img;window_size=41,properties=[GLCM_Optimizado.HARALICK_ENTROPY,GLCM_Optimizado.HARALICK_ASM]
)
   # datos=hcat(vec(val3[1]),vec(val5[1]),vec(val7[1]))
 datos=hcat(vec(val3[1]),vec(val7[1]))

for i in 2:length(val3)
       # datos=hcat(datos,vec(val3[i]),vec(val5[i]),vec(val7[i]))
        datos=hcat(datos,vec(val3[i]),vec(val7[i]))
end
return datos
end 
"""
for (i,idx) in enumerate(img)
  #  imagen2=RGB.(copy(img))
if mask[i] 
imagen2[i]=color[res2[j]]
j=j+1
else
imagen2[i]=RGB(0.0,0,0)

end
end
"""
function arboles(modelo1,modelo2,img,clase1)

colores=[RGB(1,0,0),RGB(0,1,0),RGB(0,0,1),RGB(1,1,0),RGB(1,0,1),RGB(0,1,1),RGB(1,0.5,0.5)]
img=Float64.(Gray.(img))
data=GLCM_vent(img,funcs=funcs)

println("Obteniendo clasificaciones del modelo 1")
res1=predict(modelo1,data)
mask= res1 .==clase1

println("Obteniendo clasificaciones del modelo 2")
resi2=predict(modelo2,data[mask,:])

imagen1=[color[res1[idx]] for (idx,i) in enumerate(img)]
    println("imagen 1...")

j=1
img3=RGB.(copy(img))

for (i,idx) in enumerate(img)
if mask[i] 
    img3[i]=color[resi2[j]]
    j=j+1
else
    img3[i]=RGB(0.0,0,0)
end
end

return imagen1,img3
end

## Caracterisitcas mas exitosas hasta ahorita son Entropia y IDM
# explorar si con solo 2 ventanas diferentes
