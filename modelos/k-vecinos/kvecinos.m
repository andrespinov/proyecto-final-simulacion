function kvecinos(X, Y, Xtest, Ytest)
%KVECINOS entrena, valida y testea un conjunto de muestras de clasificaci�n
%con el m�delo de K-Vecinos.
%
%	Descripci�n
%	KVECINOS(X, Y, Xtest, Ytest) 
%

    porcentaje = 0.7; %Porcentaje de la partici�n
    
    %Se definen los par�metros a estimar
    k = [1,2,3,4,5,6,7,8,9,10];
    
    for w = 1:3
        for i = 1:5
            Texto = strcat('Iteraci�n i = ', {' '}, num2str(i));
            disp(Texto);
            % Se hace la partici�n entre los conjuntos de entrenamiento y validaci�n.
            [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X, Y, porcentaje);

            %Iteraci�n por cada valor del hiper-par�metro a estimar
            for j=1:length(k)

                % Normalizaci�n de los conjuntos
                [Xtrain, mu, sigma] = zscore(Xtrain);
                Xval = normalizar(Xval, mu, sigma);

                % Generaci�n de la predicci�n del conjunto de validaci�n con
                % respecto al de entrenamiento
                Yesti = entrenamientoKVecinos(Xval, Xtrain, Ytrain, k(j));

                % Se encuentra la eficiencia de clasificaci�n
                Eficiencia(j) = (sum(Yesti == Yval))/length(Yval);
                Texto = strcat('K = ', {' '}, num2str(k(j)));
                disp(Texto);
                Texto = strcat('Eficiencia = ',{' '},num2str(Eficiencia(j)));
                disp(Texto);
            end
            %Se guarda el �ndice del par�metro con mayor eficiencia
            [~,ind] = max(Eficiencia);
            vec(i) = ind;
        end
        kEstimado = k(mode(vec));
        vec2(w) = kEstimado;
        Texto = strcat('La mejor eficiencia fue para k = ',{' '}, num2str(kEstimado));
        disp(Texto);

        % Normalizaci�n de los conjuntos
        [X, mu, sigma] = zscore(X);
        Xtest = normalizar(Xtest, mu, sigma);

        % Se entrena el modelo nuevamente con todo el conjunto para evaluar el 
        % conjunto de test con el k ganador
        Yesti = entrenamientoKVecinos(Xtest, X, Y, kEstimado);

        % Se calcula nuevamente la eficiencia para este �ltimo entrenamiento
        Eficiencia = (sum(Yesti == Ytest))/length(Ytest);
        Texto = strcat('Para el conjunto de testeo se obtuvo eficiencia = ',{' '},num2str(Eficiencia));
        disp(Texto);
    end
    kFinal = mode(vec2);
    Texto = strcat('La mejor eficiencia final fue para ks = ',{' '}, num2str(kFinal));
    disp(Texto);
end
