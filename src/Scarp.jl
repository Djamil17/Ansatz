

E(x,y,z)=(2x^2+y^2+z^2)/4

X=[1,2,3]
Y=X
Z=X

function f()
    list=[]
    for z in Z 
        for y in Y 
            for x in X
                Eᵢ=E(x,y,z)
                append!(list ,Eᵢ)
            end 
        end 
    end 
    return list 
end 