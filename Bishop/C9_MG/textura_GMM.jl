
include("MixtureGaussian_OPTIMIZED.jl")

using Images, ImageView, ImageFeatures,Statistics, .MixtureGaussianOptimized


function c_par_std(parche)


        mascara=glcm(parche,1,[0,45,90,135])

        entropia=glcm_prop.(mascara,entropy)
        idm=glcm_prop.(mascara,IDM)
        contraste=glcm_prop.(mascara,contrast)
        asm=glcm_prop.(mascara,ASM)

        e1=mean(entropia)
        e2=std(entropia)
        i1=mean(idm)
        i2=std(idm)
        c1=mean(contraste)
        c2=std(contraste)
        a1=mean(asm)
        a2=std(asm)

        # mascara=mapwindow(c_par_std,img,(7,7))

        return e1,e2,i1,i2,c1,c2,a1,a2
end

function datos_img(car)

matriz_8xNM = reshape(reinterpret(eltype(eltype(car)), car), 8, :)

# 2. Transponemos la matriz para obtener el tamaño (N*M) x 8
matriz_resultado = copy(matriz_8xNM')

return matriz_resultado

end
#img alguna imagen

function pipeline(img,clases)
	mascara=mapwindow(c_par_std,img,(7,7))
	datos=datos_img(mascara)
	modelo,_=fit_gmm(datos,clases)
	res=predict(modelo,datos)
return mascara,modelo,res
end


function arbol(char,mask_bin,sub_clases)

	modelo,_=fit_gmm(char[vec(bin),:],sub_clases)
	res=predict(modelo,char[vec(bin),:])
	return modelo,res
end

function visualizacion(img,res,clases)

	color=rand(RGB,clases)
	img_res=[color[res[idx]] for (idx,i) in enumerate(img)]
	return img_res

end


function mascara_clase(img,res,clase)
img_res=[res[idx] == clase for (idx,i) in enumerate(img)]
end




function tree(img,bin,vec,clases)

	color=rand(RGB,clases)
        n_img=RGB.(copy(img))
j=1
for (idx,i) in enumerate(img)
                if bin[idx]==0
                n_img[idx]=RGB(0,0,0)
        else
                n_img[idx]=color[vec[j]]
        j += 1
        end
        
end 
return n_img
end
