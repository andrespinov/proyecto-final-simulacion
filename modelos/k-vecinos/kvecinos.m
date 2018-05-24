function kvecinos(X, Y, Xtest, Ytest)
%KVECINOS entrena, valida y testea un conjunto de muestras de clasificación
%con el módelo de K-Vecinos.
%
%	Descripción
%	KVECINOS(X, Y, Xtest, Ytest) 
%

    porcentaje = 0.7; %Porcentaje de la partición
    
    %Se definen los parámetros a estimar
    k = [1,2,3,4,5,6,7,8,9,10];
    
    for w = 1:3
        for i = 1:5
            Texto = strcat('Iteración i = ', {' '}, num2str(i));
            disp(Texto);
            % Se hace la partición entre los conjuntos de entrenamiento y validación.
            [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X, Y, porcentaje);

            %Iteración por cada valor del hiper-parámetro a estimar
            for j=1:length(k)

                % Normalización de los conjuntos
                [Xtrain, mu, sigma] = zscore(Xtrain);
                Xval = normalizar(Xval, mu, sigma);

                % Generación de la predicción del conjunto de validación con
                % respecto al de entrenamiento
                Yesti = entrenamientoKVecinos(Xval, Xtrain, Ytrain, k(j));

                % Se encuentra la eficiencia de clasificación
                Eficiencia(j) = (sum(Yesti == Yval))/length(Yval);
                Texto = strcat('K = ', {' '}, num2str(k(j)));
                disp(Texto);
                Texto = strcat('Eficiencia = ',{' '},num2str(Eficiencia(j)));
                disp(Texto);
            end
            %Se guarda el índice del parámetro con mayor eficiencia
            [~,ind] = max(Eficiencia);
            vec(i) = ind;
        end
        kEstimado = k(mode(vec));
        vec2(w) = kEstimado;
        Texto = strcat('La mejor eficiencia fue para k = ',{' '}, num2str(kEstimado));
        disp(Texto);

        % Normalización de los conjuntos
        [X, mu, sigma] = zscore(X);
        Xtest = normalizar(Xtest, mu, sigma);

        % Se entrena el modelo nuevamente con todo el conjunto para evaluar el 
        % conjunto de test con el k ganador
        Yesti = entrenamientoKVecinos(Xtest, X, Y, kEstimado);

        % Se calcula nuevamente la eficiencia para este último entrenamiento
        Eficiencia = (sum(Yesti == Ytest))/length(Ytest);
        Texto = strcat('Para el conjunto de testeo se obtuvo eficiencia = ',{' '},num2str(Eficiencia));
        disp(Texto);
    end
    kFinal = mode(vec2);
    Texto = strcat('La mejor eficiencia final fue para ks = ',{' '}, num2str(kFinal));
    disp(Texto);
end

