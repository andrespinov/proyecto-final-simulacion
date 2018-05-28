function kvecinos(X, Y, Xtest, Ytest)
%KVECINOS entrena, valida y testea un conjunto de muestras de clasificación
%con el módelo de K-Vecinos.
%
%	Descripción
%	KVECINOS(X, Y, Xtest, Ytest) 
%

    porcentaje = 0.7; %Porcentaje de la partición
    numClases = length(unique(Ytest));
    folds = 10;
    %Se definen los parámetros a estimar
    k = [1,2,3,4,5,6,7,8,9,10];
    
    % Normalización de los conjuntos
    [X, mu, sigma] = zscore(X);
    Xtest = normalizar(Xtest, mu, sigma);
    
    for w = 1:1
        Texto = strcat('Iteración w = ', {' '}, num2str(w));
        disp(Texto);
        for i = 1:length(k)
            Texto = strcat('K = ', {' '}, num2str(k(j)));
            disp(Texto);

            for j = 1:folds
                % Se hace la partición entre los conjuntos de entrenamiento y validación.
                [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X, Y, porcentaje);
            
                % Generación de la predicción del conjunto de validación con
                % respecto al de entrenamiento
                Yesti = entrenamientoKVecinos(Xval, Xtrain, Ytrain, k(i));

                % Se encuentra la eficiencia de clasificación
                Eficiencia(j) = (sum(Yesti == Yval))/length(Yval);
            end
            % Se calcula la eficiencia de K en base a la media de todas las
            % iteraciones
            vec(i) = mean(Eficiencia);
            Texto = strcat('La eficiencia general es = ', {' '}, num2str(vec(i)));
            disp(Texto);
        end
        [~,ind] = max(vec);
        kEstimado = k(ind);
        vec2(w) = kEstimado;
        Texto = strcat('La mejor eficiencia fue para k = ',{' '}, num2str(kEstimado));
        disp(Texto);

        % Normalización de los conjuntos
        [X, mu, sigma] = zscore(X);
        Xtest = normalizar(Xtest, mu, sigma);

        % Se entrena el modelo nuevamente con todo el conjunto para evaluar el 
        % conjunto de test con el k ganador
        Yesti = entrenamientoKVecinos(Xtest, X, Y, kEstimado);

        % Se calcula la matriz de confusión para calcular las eficiencias
        % correspondientes al parámetro ganador
        MatrizConfusion = zeros(numClases, numClases);
        for m = 1:size(Xtest,1)
            MatrizConfusion(Yesti(m),Ytest(m)) = MatrizConfusion(Yesti(m),Ytest(m)) + 1;
        end
        Eficiencia = sum(diag(MatrizConfusion))/sum(MatrizConfusion(:));
        Texto = strcat('Para el conjunto de testeo se obtuvo eficiencia general = ',{' '},num2str(Eficiencia));
        disp(Texto);
        disp('Eficiencias para cada clase: ');
        for m = 1:numClases
            Texto = strcat('Clase', {' '}, num2str(m));
            disp(Texto);
            Texto = strcat ('Eficiencia de productor = ', {' '}, num2str(MatrizConfusion(m,m)/sum(MatrizConfusion(:,m))));
            disp(Texto);
            Texto = strcat ('Eficiencia de usuario = ', {' '}, num2str(MatrizConfusion(m,m)/sum(MatrizConfusion(m,:))));
            disp(Texto);
        end
    end
end

