# Capitulo nueve del Bishop
#

using LinearAlgebra
mutable struct Gaussian{T,S}
	mu::T
	sigma::S
	tam::Int
# Constructor para distribucion normal multivariada
function Gaussian(mu::T,sigma::S) where T<:AbstractVector where S<:AbstractMatrix
	tam=length(mu)
	if size(sigma, 1) != size(sigma, 2)
            error("La matriz de covarianza (sigma) debe ser cuadrada")
        end
	new{T,S}(mu,sigma,tam)

end
#Contructor para distribucion normal de una variable
function Gaussian(mu::T,sigma::S) where T<:Real where S<:Real
	new{T,S}(mu,sigma,1)

end

end
"""
funcion que evalua un vector de variables de la distribucion
"""
function (distribucion::Gaussian)(xargs...)
	factor1=1/(2*pi)^(distribucion.tam/2)
	factor2=1/(det(distribucion.sigma)^0.5)

	mius=xargs .- distribucion.mu
	inversa=pinv(distribucion.sigma)

	e=exp(-0.5* mius' * inversa * mius)

	return factor1*factor2*e
end
function (distribucion::Gaussian)(x)
	factor=1/(2*pi*(distribucion.sigma^2))^0.5

	mius=x - distribucion.mu

	e=exp(-(1/(2*distribucion.sigma^2)) * mius^2)

	return factor*e
end
struct MG{T}
	cantidad::Int
	distribuciones::Vector{Gaussian}
	pi::Vector{T}
end



function MG(num_distribuciones,datos;clip=3)
	dim=size(datos,2)
	tam=size(datos,1)
	println("dimensiones: ",dim," tamaño: ",tam)

	distribuciones,pi=inicializacion(dim,tam,num_distribuciones)
	println("medias y sigmas de lasa distribuciones:")
	for i in distribuciones
println("mu: ", i.mu)
		println("sigma: ",i.sigma)
	end
	#matriz de responsabilidades que seria una matriz n x k.
	
	gamma=Matrix{Float64}(undef,tam,num_distribuciones)
	log_anterior=0
	log_like=1

while round(log_anterior, digits=clip) != round(log_like, digits=clip)
	log_anterior=log_like

	for n in 1:tam
		for k in 1:num_distribuciones
		gamma[n,k]=responsabilidad(distribuciones,pi,datos[n,:],k)
		end 
	end
	N=sum(gamma,dims=1) #vector con k entradas

	for k in 1:num_distribuciones
		mus!(distribuciones[k],datos,gamma[:,k],N[k])
		sigmas!(distribuciones[k],datos,gamma[:,k],N[k])
		pi[k]=pis(N[k],tam)
	end

	log_like=0
	t=0
	for n in 1:tam
		for k in 1:num_distribuciones
			t += pi[k]* distribuciones[k](datos[n,:]...)
		end
		log_like += log(t)
	end

		println("calculando... log: ",log_like)
	end
	return distribuciones
end

function inicializacion(dim,tam,num)
# Inicializacion de los datos
	medias_iniciales=rand(num,dim)
	A=rand(dim,dim)
	sigmas_iniciales= [(A * A') + 1e-6 * I for _ in 1:num]

	
	distribuciones=[Gaussian(medias_iniciales[i,:],sigmas_iniciales[:,:][i]) for i in 1:num]

	_pi = rand(num)
	pi_inicial = _pi ./ sum(_pi)

return distribuciones,pi_inicial

end

function mus!(dist,datos,gammas,N)
	f=1/N
	prod=datos' *gammas
	dist.mu=f .* prod
end

function sigmas!(dist,datos,gammas,N)
	f=1/N
	distancia=datos' .- dist.mu
	suma=distancia * (distancia' .* gammas)
	epsilon = 1e-6

	dist.sigma=f .* suma +(epsilon * I)
end

function pis(N,tam)
	return N/tam
end


"""
gamma de la variable latente z en la posicion k

quiza tambien x es el valor de la marginalizacion en esa dimension?

o codificacion de la variable x a su espacio latente z
"""
function responsabilidad(distribuciones,pi,x,k)
	## esto se calcula de mas, optimizar
	t=0
	for i in 1:length(distribuciones)
		t += pi[i]* distribuciones[i](x...)
	end
	##

	nominador=pi[k]*distribuciones[k](x...)

	return nominador/t
end


valores=[RGB(i.mu...) for i in p]
vectores=[eigvecs(i.sigma) for i in p]
function direccion(x,p,vectores,valores;cant=0.1)
	b=[red(x),blue(x),green(x)]
	vals=[ i(b...) for i in p]
	idx=argmax(vals)
	prob=vals[idx]/sum(vals) *cant
	n_val=valores[idx] .+ RGB((vectores[idx][:,1] .* prob)...) 

	return n_val
end
function agrupado(x,p,val)
	b=[red(x),blue(x),green(x)]
	valores=[ i(b...) for i in p]
	idx=argmax(valores)

	return val[idx]
end
