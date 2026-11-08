# Segmentación y Detección de Amibas con GLCM y HGMM

Este repositorio contiene la implementación, entrenamiento y resultados de inferencia del **Modelo de Mezclas Gaussianas Jerárquico (HGMM)** combinado con descriptores de textura **GLCM (Gray-Level Co-occurrence Matrix)** para la segmentación de imágenes de amibas microscópicas.

---

## 1. Estructura del Repositorio

- **`segmentaciones/`**: Contiene las **90 imágenes** segmentadas mediante el modelo HGMM jerárquico completo (utilizando una paleta consistente de 8 clases para hojas y subcomponentes).
- **`deteccion_gaussiana/`**: Contiene las **90 imágenes** procesadas con la función de detección de una sola Gaussiana (`diferenciar_amiba_gaussiana`), resaltando en rojo los píxeles clasificados como amiba.
- **`comparativas/`**: Contiene las **90 imágenes** comparativas lado a lado:  
  `[Imagen Original | Segmentación HGMM | Detección Gaussiana Única]`.
- **`GLCM_Jerarquico.jl`**: Código fuente principal que define la topología del árbol, extracción de ventanas GLCM, inferencia jerárquica y la función `diferenciar_amiba_gaussiana`.
- **`GLCM_Optimizado.jl`**: Módulo optimizado de cómputo GLCM multiescala y multiángulo.
- **`Bishop/`**: Implementación base de modelos de mezclas gaussianas y topología de árbol (`HGMM_m`).
- **`ejecutar_todo.jl`**: Script automatizado que entrena el modelo en `ameba2507_0008.jpg` y procesa todo el lote de imágenes.

---

## 2. Entrenamiento del Modelo

El modelo HGMM fue entrenado siguiendo las instrucciones del archivo `GLCM_Jerarquico.jl`:
- **Imagen de entrenamiento**: `ameba2507_0008.jpg` (tamaño 512 × 640).
- **Características**: Vector de 9 características GLCM (propiedad de Contraste calculada en ventanas de tamaños 3, 11 y 19 píxeles, sobre ángulos horizontales, diagonales y verticales).
- **Topología del árbol**: `TreeTopology([0, 1, 1, 2, 2, 3, 3])`
  - Nivel 1: Nodo raíz (1).
  - Nivel 2: Nodos intermedios (2 y 3).
  - Nivel 3: Hojas (4, 5, 6 y 7), cada una con un GMM de 2 subcomponentes.

---

## 3. Identificación de la Componente de la Amiba

Al cruzar las segmentaciones con las anotaciones de verdad fundamental (bounding boxes de YOLO):
- La **amiba** corresponde a la **Hoja 6, Componente 2** del HGMM.
- En la paleta de colores de segmentación se representa en color **amarillo brillante** (`RGB(1.0, 1.0, 0.455)`).
- El fondo y debris corresponden predominantemente a la hoja 5 (componente 1 en gris oscuro) y hoja 4.

---

## 4. Detección con una Sola Gaussiana

Se implementó la función `diferenciar_amiba_gaussiana`:

```julia
diferenciar_amiba_gaussiana(img, modelo::HGMM_m.HGMM; hoja=6, comp=2, umbral=-35.0, vals=nothing)
```

Esta función extrae los descriptores GLCM de cada píxel y evalúa **únicamente la densidad de probabilidad (logpdf)** de la componente Gaussiana de la amiba (`modelo.gmms[6].components[2]`), evitando recorrer todo el árbol jerárquico.

### ¿Se puede diferenciar la amiba únicamente con esta Gaussiana?
**Sí, con notable claridad.**
- **Log-verosimilitud dentro de la amiba**: Mediana = **-35.98** (rango intercuartil: [-52.67, -29.37]).
- **Log-verosimilitud en el fondo**: Mediana = **-101.35** (rango intercuartil: [-102.02, -100.12]).
- **Diferencia de medianas**: **> 65 unidades logarítmicas**.
- Con un umbral entre **-35.0 y -40.0**, la Gaussiana aislada aísla la morfología de la amiba con un balance óptimo entre sensibilidad y especificidad, validando la hipótesis.
