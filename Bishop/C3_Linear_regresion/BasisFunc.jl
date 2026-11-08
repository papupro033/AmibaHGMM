

using LinearAlgebra, Plots, DataFrames,Statistics

"""
Matriz phi.

Matriz NxM donde N es el numero de variables y M el numero de de funciones base.

En este caso x puede tener varias variables
"""
function phi(func,x)
	if typeof(x) <:AbstractVector
		matriz=[f(v...) for f in func, v in x]
	else
		matriz=[f(x...) for f in func]
	end
return matriz
end

"""
Estructura de datos que tiene los datos necesarios para construir un modelo de basis functions.

"""
mutable struct ModeloBayesiano
func
m
S
alpha
beta
	function ModeloBayesiano(alpha,beta,func)
		s_init=inv(Matrix(LinearAlgebra.I,length(func),length(func))).*alpha
		m_init=zeros(length(func))
		new(func,m_init,s_init,alpha,beta)
	end
end


"""
funcion de entrenamiento para el modeloBayesiano
"""
function entreno!(m::ModeloBayesiano,x,t;lim=5)

	p=phi(m.func,x)
	prev_beta=m.beta
	prev_alpha=m.alpha

	while(true)
	s_n=m.S+ m.beta*(p * p')
	m_n=pinv(s_n)*(m.S * m.m + m.beta*(p*t))
	# alpha y beta optimo
	
	mat=m.beta* p * p'
	eig=eigvals(mat)
	gamma=sum([i/(m.alpha+i) for i in eig])

	m.alpha=gamma/(m_n' * m_n)
	b=(1/(length(x)-gamma))*sum((t .- (m_n' *p)) .^2)
	m.beta=1/(b)
	m.S=s_n
	m.m=m_n
	print(m.beta)
		(trunc(prev_beta,digits=lim)==trunc(m.beta,digits=lim) && trunc(prev_alpha,digits=lim) == trunc(m.alpha,digits=lim)) && break
	prev_beta=m.beta 
	prev_alpha=m.alpha
	end
end


"""
Funcion de infernecia para el modelo m. ya entrenado

"""

function inference(m::ModeloBayesiano,x)
	p=phi(m.func,x)
	mu=only(p'*m.m)
	sigma=1/m.beta+ only(p'* m.S *p)
	return mu, sigma
end

# Funciones base para varios variables
#
# Las funciones tambien pueden ser propias
#
"""
funciones para inferencia financiera
"""
function linear1(x,y,z,w,v)
	suma=x*y
	return suma
end
function linear2(x,y,z,w,v)
	suma=z*w
	return suma
end
function linear3(x,y,z,w,v)
	suma=sum([x*y,y*z,z*w,w*v,x*z,x*w,x*v])
	return suma
end
function linear4(x,y,z,w,v)
	suma=sum([x*y,x*z,x*w,x*v])
	return suma
end

linear=[linear1,linear2,linear3,linear4]

function sigmoide(x,y,z,w,v)
	t1=1/(1+exp(-x))
	t2=1/(1+exp(-y))
	t3=1/(1+exp(-z))
	t4=1/(1+exp(-w))
	t5=1/(1+exp(-v))
	return sum([t1,t2,t3,t4,t5])
end

function cossin(x,y,z,w,v)
	t1=cos(x*y)+sin(x*y)
	t2=cos(z*y)+sin(z*y)
	t3=cos(z*w)+sin(z*w)
	t4=cos(w*v)+sin(w*v)
	t5=cos(v*x)+sin(x*v)

	return t1*t2*t3*t4*t5

end

function rescale(img)
(lo, hi) = extrema(img)
        rescaled_img = ((img .- lo) ./ (hi - lo))

print("low: ",lo,"high: ",hi)
end

function denormalize(val_norm, min_val, max_val)
    return val_norm * (max_val - min_val) + min_val
end

function acondicionar(data)
transform!(df, names(df, Number) .=> rescale .=> names(df, Number))
	vec=[Tuple(i) for i in eachrow(data[:,3:7])]

 return vec
end



function extract_stock_features(df::DataFrame, n::Int)
    # Total de filas en el DataFrame
    N = nrow(df)
    
    # Extraemos columnas manejando posibles valores 'missing' iniciales
    adj_close   = coalesce.(df[!, Symbol("Adj Close_WALMEX.MX")], 0.0)
    log_returns = coalesce.(df[!, "log_return"], 0.0)
    high        = coalesce.(df[!, Symbol("High_WALMEX.MX")], 0.0)
    low         = coalesce.(df[!, Symbol("Low_WALMEX.MX")], 0.0)
    close       = coalesce.(df[!, Symbol("Close_WALMEX.MX")], 0.0)
    volume      = coalesce.(df[!, Symbol("Volume_WALMEX.MX")], 0.0)

    # 1. Media Móvil del Precio Ajustado (runmean ya respeta la longitud original)
    sma_adj_close = runmean(adj_close, n)

    # 2. Volatilidad Móvil (runstd ya respeta la longitud original)
    volatility = runstd(log_returns, n)

    # 3. Spread Intradía Normalizado (Varianza dentro del mismo día)
    intraday_spread = (high .- low) ./ (close .+ 1e-8)

    # 4. Volumen Relativo (Picos de interés)
    volume_sma = runmean(volume, n)
    relative_volume = volume ./ (volume_sma .+ 1e-8)

    # 5. NUEVO FEATURE: Momentum del Precio (Distancia porcentual a la media móvil)
    price_to_sma = (adj_close .- sma_adj_close) ./ (sma_adj_close .+ 1e-8)

    # --- Función auxiliar para Normalizar (Z-score) y limpiar NaNs ---
    function normalize_and_clean(v::Vector{Float64})
        # Filtramos los NaN o Infinitos para calcular estadísticas reales
        valid_mask = isfinite.(v)
        valid_vals = v[valid_mask]
        
        # Si no hay valores válidos, regresamos ceros
        if isempty(valid_vals)
            return zeros(Float64, N)
        end
        
        mu = mean(valid_vals)
        sigma = std(valid_vals)
        sigma = sigma == 0 ? 1.0 : sigma # Evitar división por cero
        
        # Aplicamos la normalización y reemplazamos los NaN/Inf con 0.0 (la media)
        norm_v = zeros(Float64, N)
        for i in 1:N
            if valid_mask[i]
                norm_v[i] = (v[i] - mu) / sigma
            else
                norm_v[i] = 0.0
            end
        end
        return norm_v
    end

    # Aplicamos la limpieza y normalización a todas las características
    f1 = normalize_and_clean(sma_adj_close)
    f2 = normalize_and_clean(volatility)
    f3 = normalize_and_clean(intraday_spread)
    f4 = normalize_and_clean(relative_volume)
    f5 = normalize_and_clean(price_to_sma)

    # --- Construcción del Array de salida ---
    # Creamos un Vector de Vectores del tamaño del DataFrame
    result = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        result[i] = [f1[i], f2[i], f3[i], f4[i], f5[i]]
    end

    return result
end
"""
Codigo Ejemplo

using DataFrames,CSV,Plots

include("BasisFunc.jl")

df=CSV.read("/home/lupiita/Proyectos/Bishop/Data_sets/WALMEX.MX_datos_bolsa.csv",DataFrame)


m=ModeloBayesiano(0.0,2.0,[linear...,sigmoide,cossin])

t=df[!,"log_return"]
"""
