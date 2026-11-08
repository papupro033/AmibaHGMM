
function vectorizar(args...)
	datos=vec(args[1])
	for i in args[2:end]
	datos=hcat(datos,vec(i))

end
	return datos

end

using Wavelets
using WaveletsExt
using ImageFiltering
using LinearAlgebra

"""
    extract_pixel_embeddings_swt(img::Matrix{Float64}; levels::Int=2, window_size::Int=3)

Genera un tensor M x N x Feature_Dim con los coeficientes SWT multinivel y su 
energía local promediada en un entorno (window_size x window_size).
"""
function extract_pixel_embeddings_swt(img::Matrix{Float64}; levels::Int=2, window_size::Int=3)
    M, N = size(img)
    
    # 1. Filtros Haar para SWT (algoritmo à trous)
    h = [1.0, 1.0] / sqrt(2)
    g = [-1.0, 1.0] / sqrt(2)
    
    # Función auxiliar para interpolar ceros en filtros
    upsample(f, lvl) = begin
        step = 2^(lvl - 1)
        f_up = zeros(Float64, (length(f)-1)*step + 1)
        for k in 1:length(f); f_up[(k-1)*step + 1] = f[k]; end
        f_up
    end

    feature_maps = Matrix{Float64}[]
    approx = copy(img)

    # 2. Descomposición SWT recursiva preservando M x N
    for lvl in 1:levels
        h_up = upsample(h, lvl) .* (1/sqrt(2))
        g_up = upsample(g, lvl) .* (1/sqrt(2))

        LL = imfilter(approx, centered(h_up * h_up'), "reflect")
        LH = imfilter(approx, centered(h_up * g_up'), "reflect")
        HL = imfilter(approx, centered(g_up * h_up'), "reflect")
        HH = imfilter(approx, centered(g_up * g_up'), "reflect")

        # Guardar mapas de detalle del nivel actual
        push!(feature_maps, LH)
        push!(feature_maps, HL)
        push!(feature_maps, HH)

        approx = LL
    end
    # Agregar aproximación del último nivel
    pushfirst!(feature_maps, approx) # [LL_L, LH_1, HL_1, HH_1, ...]

    num_maps = length(feature_maps) # 3L + 1
    
    # 3. Suavizado local opcional (Energía Local) para reducir ruido en el píxel
    kernel = ones(window_size, window_size) ./ (window_size^2)
    
    # Tensor de salida: M x N x (2 * num_maps)
    # Guardamos: [Valor_Coeficiente, Energia_Local] por cada mapa
    pixel_features = zeros(Float64, M, N, num_maps * 2)

    for idx in 1:num_maps
        fmap = feature_maps[idx]
        energy_local = imfilter(fmap .^ 2, kernel)

        pixel_features[:, :, (idx - 1)*2 + 1] = fmap
        pixel_features[:, :, (idx - 1)*2 + 2] = energy_local
    end

    return pixel_features # M x N x D
end

# === Uso ===
img = rand(128, 128) # Imagen 128x128
# Genera características por píxel usando 2 niveles de SWT
features_tensor = extract_pixel_embeddings_swt(img, levels=2, window_size=3)

M, N, D = size(features_tensor)
println("Dimensiones de la imagen: ", M, "x", N)
println("Dimensión del vector por píxel: ", D) # (3*2 + 1) * 2 = 14 características

# Para obtener el vector de características del píxel en (row=50, col=30):
pikel_vec = features_tensor[50, 30, :]
