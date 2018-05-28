function Modelo = entrenarSVM(X, Y, boxConstraint, sigma, tipoKernel)
    
    if (tipoKernel == 1) % Kernel Lineal
        Modelo = trainlssvm({X, Y, 'c', boxConstraint, [], 'lin_kernel'});
    else % Kernel Gaussiano
        Modelo = trainlssvm({X, Y, 'c', boxConstraint, sigma, 'RBF_kernel'});
    end
end