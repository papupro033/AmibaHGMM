

module LinearBasisFunctionModels
"""md
# Linear Basis Fucntion models 

utiliza funcionaes para hacer una regresion lineal para funciones con comportamiento no lineal.

# 3.1.1 Maximum likelihood y least squares
#

"""
#Cargar libreria
using LinearAlgebra


# Basis functions

abstract type BasisFunction <: Function end

struct SigmoidBasis <: BasisFunction end

(::SigmoidBasis)(x)= 1/(1+exp(-x))
struct SinBasis <: BasisFunction 
omega
end

(p::SinBasis)(x)= sin(p.omega * x)
struct CosBasis <: BasisFunction
omega
end

(p::CosBasis)(x)= cos(p.omega * x)



struct GaussianBasis{T} <: BasisFunction 
	mu::T
	s::T
end

(g::GaussianBasis)(x)= exp(-((x-g.mu)^2)/(2*g.s^2))

struct PolynomialBasis{T} <: BasisFunction 
	grado::Int
	coeff::Vector{T}
	function PolynomialBasis{T}(vec) where T
		grado=length(vec)
		new(grado, Vector{T}(vec)) 
    	end
end

function PolynomialBasis(coeffs::Vector{T}) where T
    PolynomialBasis{T}(coeffs)
end

(p::PolynomialBasis)(x) = evalpoly(x,p.coeff)

export PolynomialBasis,TanhBasis,GaussianBasis
# Estructura de datos de inferencia

struct Modelo{F<:BasisFunction,T<:Number}
	func::Vector{F}
	w::Vector{T}
	beta::T
	function Modelo(x::AbstractVector{T},t::AbstractVector{T}, func::Vector{F}) where {T<:Number, F<:BasisFunction}
		matriz = Matrix{T}(undef,length(x),length(func))
	  for (idx, f) in enumerate(func)
			matriz[:,idx] .= f.(x)
          end
		println(matriz)
		w=w_ml(matriz,t)
		print(w)
		b=beta(w,matriz,t)
		new{F,Float64}(func,w,b)
	end
end

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

function entreno(m::ModeloBayesiano,x,t)
	p=phi(m.func,x)
	S_n=m.S+ m.beta*(p' * p)
	m_n=pinv(S_n)*(m.S * m.m + m.beta*(p'*t))

	m.S=S_n
	m.m=m_n
end
export Modelo,ModeloBayesiano
# Matriz de diseño
#
function w_ml(mat,t)

	p1=mat' *mat
	p2=mat' *t
	w=inv(p1)*p2
return w

end
function beta(w,mat,t)
	p1=t .- ( mat* w)
	p2=sum(p1 .^2)
	n=1/length(p1)
	salida=n*p2
	return 1/salida
end
	
function inference(m::Modelo,x)
	res=[i(x) for i in m.func]
	t=m.w' * res
end

function inference(m::ModeloBayesiano,x)

	p=phi(m.func,x)
	mu=only(p*m.m)
	sigma=1/m.beta+ only(p* m.S *p')
	return mu, sigma
end

function phi(func,x)
	matriz = Matrix(undef,length(x),length(func))
	  for (idx, f) in enumerate(func)
			matriz[:,idx] .= f.(x)
          end
return matriz

end

export inference,entreno
end

