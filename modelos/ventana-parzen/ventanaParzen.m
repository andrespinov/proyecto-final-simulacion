function ventanaParzen(X, Y, Xtest, Ytest)
    
    porcentaje = 0.7; %Porcentaje de la partición
    numClases = length(unique(Ytest));
    
    % Normalización de los conjuntos
    [X, mu, sigma] = zscore(X);
    Xtest = normalizar(Xtest, mu, sigma);
    
    %Se definen los parámetros a estimar
    h = [0.05, 0.1, 0.5, 1, 10];
    for w = 1:1
        Texto = strcat('Iteración w = ', {' '}, num2str(w));
        disp(Texto);
        for i = 1:5
            Texto = strcat('Iteración i = ', {' '}, num2str(i));
            disp(Texto);
            % Se hace la partición entre los conjuntos de entrenamiento y validación.
            [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X,Y, porcentaje);

            %Iteración por cada valor del hiper-parámetro a estimar
            for j = 1:length(h)

                % Partición del conjunto de entrenamiento para la generación de las
                % fdp de cada clase
                [Xtrain1, Xtrain2, Xtrain3, Xtrain4] = separarClases(Xtrain, Ytrain);

                % Generación de las fdp de cada clase
                funcion1 = entrenamientoParzen(Xval, Xtrain1, h(j));
                funcion2 = entrenamientoParzen(Xval, Xtrain2, h(j));
                funcion3 = entrenamientoParzen(Xval, Xtrain3, h(j));
                funcion4 = entrenamientoParzen(Xval, Xtrain4, h(j));

                funcion = [funcion1, funcion2, funcion3, funcion4];
                [~,Yesti] = max(funcion, [], 2);

                % Se encuentra la eficiencia de clasificación
                Eficiencia(j) = (sum(Yesti == Yval))/length(Yval);
                Texto = strcat('H = ', {' '}, num2str(h(j)));
                disp(Texto);
                Texto = strcat('Eficiencia = ',{' '},num2str(Eficiencia(j)));
                disp(Texto);
            end
            %Se guarda el índice del parámetro con mayor eficiencia
            [~,ind] = max(Eficiencia);
            vec(i) = ind;
        end
        hEstimado = h(mode(vec));
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

