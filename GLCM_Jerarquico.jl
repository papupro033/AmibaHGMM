
using Images,ImageView
## incluir GMM jerarquico y procesador de matriz GLCM
path_HGMM="Bishop/C9_MG/HGMM_IA.jl"
    include(path_HGMM)

path_GLCM="GLCM_Optimizado.jl"
    include(path_GLCM)

using .HGMM_m
using .GLCM_Optimizado


#### Funciones para entrenamiento y inferencia
#
#
path_img="/home/lupiita/YOLO_amiba/YOLO_amiba/ima_amiba/"


funcs=[GLCM_Optimizado.HARALICK_ENTROPY,GLCM_Optimizado.HARALICK_IDM,GLCM_Optimizado.HARALICK_CONTRAST,GLCM_Optimizado.HARALICK_ASM]

funcs=[GLCM_Optimizado.HARALICK_CONTRAST]


function ventanas(img;tam=[3,11,19],propiedades=funcs)
    num_car=length(tam)*length(propiedades)

    img=Float64.(Gray.(img));
    x,y=size(img)

    angulos=[(0,π),(π/4, 3*π/4),(π/2,3*π/2)]
    img_n=[copy(img) for _ in tam , _ in propiedades, _ in angulos]
    datos=vec([0 for _ in 1:x*y])     #Matrix{Float64}(undef,x*y,num_car)

    for (tdx,theta) in enumerate(angulos)
    for (idx,i) in enumerate(tam)
            chars=glcm_hybrid_multi(img,window_size=i,angles=[theta...],properties=funcs)
        for j in 1:length(propiedades)
img_n[idx,j,tdx]=chars[j]

            datos=hcat(datos,vec(chars[j]))
        end
    end
end
    return img_n, datos[:,2:end]
end

grafo=TreeTopology([0,1,1,2,2,3,3]) # nodo con 2 clases principales , dos sub clases para cada clase principal y 2 subclases para cada subclase.
grafo1=TreeTopology([0,1,1,1,1,2,2,3,3,4,4,5,5]) # nodo con 2 clases principales , dos sub clases para cada clase principal y 2 subclases para cada subclase.

function pred_clases(img,modelo;vals=nothing)
    clases=length(modelo.topology.parents)
if isnothing(vals)
    img_n,datos=ventanas(img)
else
        datos=vals
end
    res_hoja,comp_res=predict(modelo,datos)

    componentes=length(unique(res_hoja))
    subcomponentes=length(unique(comp_res))

    colores = distinguishable_colors(componentes*subcomponentes)


    seg=[colores[(res_hoja[idx]%componentes)*subcomponentes+comp_res[idx]] for (idx,i) in enumerate(img)]


    return seg #img_n
end

function pred_clases_niveles(img, modelo; vals=nothing)
    # 1. Extraer características
if isnothing(vals)
    img_n, datos = ventanas(img)
else 
        datos=vals
end
    # 2. Predecir para todos los niveles
    asignaciones_nivel, comp_res = predict_niveles(modelo, datos)
    max_niveles = length(modelo.topology.levels)
    
    # Aquí guardaremos la segmentación de cada nivel
    segmentaciones_por_nivel = []
    
    # 3. Generar la imagen segmentada para cada nivel del grafo (nodos)
    for l in 1:max_niveles
        # Nodos a los que pertenecen los pixeles en el nivel actual
        nodos_nivel = asignaciones_nivel[l, :]
        nodos_unicos = unique(nodos_nivel)
        num_clases = length(nodos_unicos)
        
        # Generar colores (max(2, ...) es para evitar error si en la raíz solo hay 1 clase)
        colores = distinguishable_colors(max(2, num_clases))
        
        # Crear un diccionario seguro para mapear el ID del nodo a su color
        mapa_colores = Dict(nodo => colores[i] for (i, nodo) in enumerate(nodos_unicos))
        
        # Generar la imagen (arreglo de colores)
        seg_nivel = [mapa_colores[nodos_nivel[idx]] for (idx,i) in enumerate(img)]
        push!(segmentaciones_por_nivel, seg_nivel)
    end
    
    # 4. Generar la segmentación final (hojas + subcomponentes) igual a tu código original
    # Obtenemos las hojas finales (que están en el último nivel)
    res_hoja = asignaciones_nivel[max_niveles, :]
    
    # En lugar de usar aritmética de módulo, creamos pares únicos (nodo_hoja, componente)
    # Esto es mucho más seguro para evitar errores de índice (BoundsError) en Julia
    pares_unicos = unique([(res_hoja[i], comp_res[i]) for i in 1:length(res_hoja)])
    colores_finales = distinguishable_colors(length(pares_unicos))
    
    mapa_colores_final = Dict(par => colores_finales[i] for (i, par) in enumerate(pares_unicos))
    
    seg_final = [mapa_colores_final[(res_hoja[idx], comp_res[idx])] for (idx,i) in enumerate(img)]
    push!(segmentaciones_por_nivel, seg_final)
    
    # Devuelve un Vector de segmentaciones y la imagen normalizada
    # (El tamaño del vector es max_niveles + 1)

if isnothing(vals)
    return segmentaciones_por_nivel, img_n
else
    return segmentaciones_por_nivel
end
end

"""
    diferenciar_amiba_gaussiana(img, gaussiana::HGMM_m.Gaussian; umbral=-40.0, vals=nothing)
    diferenciar_amiba_gaussiana(img, modelo::HGMM_m.HGMM; hoja=6, comp=2, umbral=-40.0, vals=nothing)

Evalúa una sola componente Gaussiana (específicamente la de la hoja y componente del GMM
que modela la amiba: por defecto hoja 6, componente 2) directamente sobre las características
GLCM de los píxeles de la imagen, sin tener que recorrer todo el árbol jerárquico.

Retorna:
- `mascara`: Matriz Booleana (H, W) donde `true` indica detección de amiba.
- `log_p_mat`: Matriz Float64 (H, W) con los valores de log-verosimilitud (logpdf).
- `img_overlay`: Imagen RGB donde los píxeles detectados se superponen en rojo sobre la imagen original.
"""
function diferenciar_amiba_gaussiana(img, gaussiana::HGMM_m.Gaussian; umbral::Float64=-40.0, vals=nothing)
    h, w = size(img)
    if isnothing(vals)
        img_n, datos = ventanas(img)
    else
        datos = vals
    end

    N = size(datos, 1)
    log_probs = Vector{Float64}(undef, N)
    for n in 1:N
        log_probs[n] = HGMM_m.logpdf(gaussiana, @view datos[n, :])
    end
    log_p_mat = reshape(log_probs, h, w)
    mascara = log_p_mat .>= umbral

    gray_img = Gray.(img)
    img_overlay = RGB.(gray_img, gray_img, gray_img)
    for idx in findall(mascara)
        img_overlay[idx] = RGB(1.0, 0.0, 0.0)
    end

    return mascara, log_p_mat, img_overlay
end

function diferenciar_amiba_gaussiana(img, modelo::HGMM_m.HGMM; hoja::Int=6, comp::Int=2, umbral::Float64=-40.0, vals=nothing)
    gaussiana = modelo.gmms[hoja].components[comp]
    return diferenciar_amiba_gaussiana(img, gaussiana; umbral=umbral, vals=vals)
end


"""
img=load(path_img*"ameba2507_0008.jpg");
img1=load(path_img*"ameba2507_0006.jpg");

imagen2=load(path_img*"ameba2507_0010.jpg");
img_n,datos=ventanas(img)
modelo,logs,indices=fit_gmm(datos,grafo)

seg,img_n=pred_clases_niveles(img,modelo)
seg1,img_n1=pred_clases_niveles(img1,modelo)

vec(Float64.(p))


RGB(1.0,1.0,0.455)
RGB(0.0,0.314,0.902)
RGB{N0f8}(0.675,0.0,0.278)
"""
pdf_normal = (μ, σ, x) -> (1 / (σ * sqrt(2 * π))) * exp(-0.5 * ((x - μ) / σ)^2)
function dist(x)
    x=gray.(x)
    media=mean(x)
    desviacion=std(x)
    x_p=x[div.(size(x) .+ 1, 2)...]
    val=pdf_normal(media,desviacion,x)
return media #,desviacion,val
end
