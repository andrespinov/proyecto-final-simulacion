function ventanaParzen(X, Y, Xtest, Ytest)
    
    porcentaje = 0.7; %Porcentaje de la partici�n
    
    % Normalizaci�n de los conjuntos
    [X, mu, sigma] = zscore(X);
    Xtest = normalizar(Xtest, mu, sigma);
    
    %Se definen los par�metros a estimar
    h = [0.05, 0.1, 0.5, 1, 10];
    for w = 1:1
        Texto = strcat('Iteraci�n w = ', {' '}, num2str(w));
        disp(Texto);
        for i = 1:5
            Texto = strcat('Iteraci�n i = ', {' '}, num2str(i));
            disp(Texto);
            % Se hace la partici�n entre los conjuntos de entrenamiento y validaci�n.
            [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X,Y, porcentaje);

            %Iteraci�n por cada valor del hiper-par�metro a estimar
            for j = 1:length(h)

                % Partici�n del conjunto de entrenamiento para la generaci�n de las
                % fdp de cada clase
                [Xtrain1, Xtrain2, Xtrain3, Xtrain4] = separarClases(Xtrain, Ytrain);

                % Generaci�n de las fdp de cada clase
                funcion1 = entrenamientoParzen(Xval, Xtrain1, h(j));
                funcion2 = entrenamientoParzen(Xval, Xtrain2, h(j));
                funcion3 = entrenamientoParzen(Xval, Xtrain3, h(j));
                funcion4 = entrenamientoParzen(Xval, Xtrain4, h(j));

                funcion = [funcion1, funcion2, funcion3, funcion4];
                [~,Yesti] = max(funcion, [], 2);

                % Se encuentra la eficiencia de clasificaci�n
                Eficiencia(j) = (sum(Yesti == Yval))/length(Yval);
                Texto = strcat('H = ', {' '}, num2str(h(j)));
                disp(Texto);
                Texto = strcat('Eficiencia = ',{' '},num2str(Eficiencia(j)));
                disp(Texto);
            end
            %Se guarda el �ndice del par�metro con mayor eficiencia
            [~,ind] = max(Eficiencia);
            vec(i) = ind;
        end
        hEstimado = h(mode(vec));
        vec2(i) = hEstimado;
        Texto = strcat('La mejor eficiencia fue para h = ',{' '}, num2str(hEstimado));
        disp(Texto);

        % Se entrena el modelo nuevamente con todo el conjunto para evaluar el 
        % conjunto de test con el h ganador
        [Xtrain1, Xtrain2, Xtrain3, Xtrain4] = separarClases(X, Y);

        funcion1 = entrenamientoParzen(Xtest, Xtrain1, hEstimado);
        funcion2 = entrenamientoParzen(Xtest, Xtrain2, hEstimado);
        funcion3 = entrenamientoParzen(Xtest, Xtrain3, hEstimado);
        funcion4 = entrenamientoParzen(Xtest, Xtrain4, hEstimado);

        funcion = [funcion1, funcion2, funcion3, funcion4];
        [~,Yesti] = max(funcion, [], 2);

        % Se calcula nuevamente la eficiencia para este �ltimo entrenamiento
        Eficiencia = (sum(Yesti == Ytest))/length(Ytest);
        Texto = strcat('Para el conjunto de testeo se obtuvo eficiencia = ',{' '},num2str(Eficiencia));
        disp(Texto);
    end
    hFinal = mode(vec2);
    Texto = strcat('La mejor eficiencia final fue para h = ',{' '}, num2str(hFinal));
    disp(Texto);
end
