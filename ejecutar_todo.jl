using Images, Statistics

include("GLCM_Jerarquico.jl")

println("==========================================================")
println("1. ENTRENAMIENTO DEL HGMM")
println("==========================================================")
println("Cargando imagen de entrenamiento: ", path_img * "ameba2507_0008.jpg")
img_train = load(path_img * "ameba2507_0008.jpg")

println("Extrayendo características GLCM (ventanas)...")
local datos_train
redirect_stdout(devnull) do
    global _, datos_train = ventanas(img_train)
end
println("Dimensiones de características de entrenamiento: ", size(datos_train))

println("Ajustando HGMM con fit_gmm(datos, grafo)...")
@time modelo, logs, indices = fit_gmm(datos_train, grafo; verbose=false)
println("Entrenamiento completado exitosamente.")

println("\n==========================================================")
println("2. CONFIGURACIÓN DE COLORES Y RUTAS DE SALIDA")
println("==========================================================")
# Mapa de colores fijo y determinista para todas las 90 imágenes
mapa_colores = Dict(
    (4, 1) => RGB{N0f8}(0.886, 0.388, 0.051), # Naranja
    (4, 2) => RGB{N0f8}(0.000, 0.494, 0.000), # Verde
    (5, 1) => RGB{N0f8}(0.150, 0.150, 0.150), # Fondo gris oscuro
    (5, 2) => RGB{N0f8}(0.000, 0.827, 1.000), # Cian
    (6, 1) => RGB{N0f8}(0.450, 0.450, 0.450), # Gris medio
    (6, 2) => RGB{N0f8}(1.000, 1.000, 0.455), # Amarillo brillante (Amiba)
    (7, 1) => RGB{N0f8}(0.675, 0.000, 0.278), # Rojo oscuro
    (7, 2) => RGB{N0f8}(0.000, 0.314, 0.902)  # Azul
)

repo_dir = "/home/lupiita/Proyectos/AmibaHGMM"
dir_seg = joinpath(repo_dir, "segmentaciones")
dir_gauss = joinpath(repo_dir, "deteccion_gaussiana")
dir_comp = joinpath(repo_dir, "comparativas")

mkpath(dir_seg)
mkpath(dir_gauss)
mkpath(dir_comp)

# Obtener lista de imágenes
archivos = filter(f -> endswith(lowercase(f), ".jpg") || endswith(lowercase(f), ".png"), readdir(path_img))
sort!(archivos)
total_imgs = length(archivos)
println("Total de imágenes encontradas en $path_img: $total_imgs")

println("\n==========================================================")
println("3. INFERENCIA PARA TODAS LAS IMÁGENES (HGMM Y GAUSSIANA ÚNICA)")
println("==========================================================")

t_inicio = time()
for (idx, archivo) in enumerate(archivos)
    t_img = time()
    ruta_in = joinpath(path_img, archivo)
    nombre_base = splitext(archivo)[1]
    
    img = load(ruta_in)
    h, w = size(img)
    
    # Extraer características GLCM en silencio
    local datos
    redirect_stdout(devnull) do
        _, datos = ventanas(img)
    end
    
    # 1. Inferencia HGMM completo
    res_hoja, comp_res = predict(modelo, datos)
    seg_img = reshape([get(mapa_colores, (res_hoja[i], comp_res[i]), RGB{N0f8}(0.5, 0.5, 0.5)) for i in 1:length(res_hoja)], h, w)
    
    # 2. Inferencia solo con la Gaussiana de la amiba (Hoja 6, Comp 2)
    # Umbral de log-verosimilitud = -35.0
    mascara, log_p_mat, overlay_gauss = diferenciar_amiba_gaussiana(img, modelo; hoja=6, comp=2, umbral=-35.0, vals=datos)
    
    # 3. Imagen comparativa lado a lado: Original | HGMM | Gaussiana Única
    img_rgb = RGB.(Gray.(img), Gray.(img), Gray.(img))
    comparativa = hcat(img_rgb, seg_img, overlay_gauss)
    
    # Guardar resultados
    save(joinpath(dir_seg, "$(nombre_base)_seg.png"), seg_img)
    save(joinpath(dir_gauss, "$(nombre_base)_gauss.png"), overlay_gauss)
    save(joinpath(dir_comp, "$(nombre_base)_comp.png"), comparativa)
    
    dt = round(time() - t_img, digits=2)
    print("\r[$(idx)/$(total_imgs)] Procesada: $archivo en $(dt)s")
    flush(stdout)
end
println()

tiempo_total = round(time() - t_inicio, digits=2)
println("Procesamiento de las $total_imgs imágenes completado en $tiempo_total segundos.")
