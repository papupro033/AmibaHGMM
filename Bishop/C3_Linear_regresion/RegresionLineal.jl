using LinearAlgebra


"""
N x M. N es por el numero de datos y M es por el numero de elementos en la regresion lineal(basis function)
"""

"""
Estructura de datos que hace la matriz phi
"""
struct PHI
funcs::AbstractVector


function PHI(fun,x)
end
end

function f_phi(func,x)
	
	if typeof(x) <:AbstractVector
		matriz=[f(v...) for v in x, f in func]
	else

		matriz=[f(v...) for v in eachrow(x), f in func]
	end
return matriz
end

"""md
# Regularizacion

a la funcion del error se lea agrega un error de pesos lambda/2 * wt * w


W es una matriz M x K, donde k son el numero de predicciones(ts).
"""
function fit(Phi::AbstractVector,x::AbstractArray,t::AbstractArray; lambda=nothing)

	phi=f_phi(Phi,x)
	M,N=size(phi)
	if isnothing(lambda)
		W=pinv(phi'*phi) * phi' * t
	else 
		W=pinv(lambda*I + phi'*phi) * phi' * t
	end
return W
end


function secuencial(w, mu,t,phi)


end
#(x,y,z)->
