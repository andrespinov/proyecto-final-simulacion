function randomforest(X,Y,Xtest,Ytest)
    
    %%% Se determina el número de clases del problema
    numClases = length(unique(Ytest)); 
    folds = 10;
    porcentaje = 0.7;
    numArboles = [10, 20, 30, 40, 50];
    
    % Normalización de los conjuntos
    [X, mu, sigma] = zscore(X);
    Xtest = normalizar(Xtest, mu, sigma);
    for w = 1:3
        Texto = strcat('Iteración w = ', {' '}, num2str(w));
        disp(Texto);
        for n = 1:length(numArboles)
            arboles = numArboles(n);
            EficienciaXarbol = zeros(arboles);
            Texto = strcat('Número de árboles = ', {' '}, num2str(arboles));
            disp(Texto);
            for fold = 1:folds
                % Se hace la partición de las muestras de entrenamiento 
                % y validación              
                [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X,Y, porcentaje);
                for i = 1:arboles

                    % Entrenamiento de los modelos.
                    Modelo = entrenarFOREST(arboles, Xtrain, Ytrain);       

                    % Validación de los modelos.      
                    Yesti = testFOREST(Modelo, Xval);
                    
                    % Se calcula la eficiencia para el árbol
                    EficienciaXarbol(i) = (sum(Yesti == Yval))/length(Yval);
                end    
                % Se obtiene la eficiencia por el fold en base a la de los
                % arboles
                EficienciaFold(fold) = mean(EficienciaXarbol);
            end
            % Se obtiene la eficiencia de la cantidad de árboles actual (n)
            Eficiencia(n) = mean(EficienciaFold);
            IC = std(EficienciaFold);
            Texto = strcat('Cantida de Arboles = ', {' '}, num2str(n));
            disp(Texto);
            Texto = strcat ('Eficiencia = ', {' '}, num2str(Eficiencia(n)), {' '}, 'IC = ', {' '}, num2str(IC));
            disp(Texto);
        end
        [~,ind] = max(Eficiencia);
        arbolesEstimado = numArboles(ind);
        EficienciaXArbol = zeros(1, arbolesEstimado);
        EficienciasXArbol = zeros(2, numClases);
        EficienciasXFold = zeros(2, numClases);
        Eficiencias = zeros(2, numClases);
        vec(w) = arbolesEstimado;
        Texto = strcat('Cantida de Arboles con mayor eficiencia fue = ', num2str(arbolesEstimado));
        disp(Texto);
        for fold = 1:folds
            for i = 1:arbolesEstimado
                % Entrenamiento del árbol.
                Modelo = entrenarFOREST(arbolesEstimado, X, Y);       
                % Validación del árbol      
                Yesti = testFOREST(Modelo, Xtest);
                
                % Se calcula la matriz de confusión para calcular las 
                % eficiencias correspondientes al parámetro ganador
                MatrizConfusion = zeros(numClases, numClases);
                for m = 1:size(Xtest,1)
                    MatrizConfusion(Yesti(m),Ytest(m)) = MatrizConfusion(Yesti(m),Ytest(m)) + 1;
                end
                EficienciaXArbol(i) = sum(diag(MatrizConfusion))/sum(MatrizConfusion(:));
                for m = 1:numClases
                    EficienciasXArbol(1,m) = EficienciasXArbol(1,m) + (MatrizConfusion(m,m)/sum(MatrizConfusion(:,m)));
                    EficienciasXArbol(2,m) = EficienciasXArbol(2,m) + (MatrizConfusion(m,m)/sum(MatrizConfusion(m,:)));
                end
            end
            % Se obtiene las eficiencias por el fold en base a la de los
            % arboles
            for m = 1:numClases
                EficienciasXFold(1,m) = EficienciasXFold(1,m) + (EficienciasXArbol(1,m)/arbolesEstimado);
                EficienciasXFold(2,m) = EficienciasXFold(2,m) + (EficienciasXArbol(2,m)/arbolesEstimado);
            end
            EficienciaFold(fold) = mean(EficienciaXArbol);
        end
        % Se obtienen las eficiencia para el conjunto de test con la
        % cantidad de árboles estimada
        EficienciaTest = mean(EficienciaFold);
        IC = std(EficienciaFold);
        Texto = strcat ('La eficiencia general para el conjunto de test con el mejor número de árboles es = ', {' '}, num2str(EficienciaTest), {' '}, 'IC = ', {' '}, num2str(IC));
        disp(Texto);
        disp('Eficiencias para cada clase: ');
        for m = 1:numClases
            Texto = strcat('Clase', {' '}, num2str(m));
            disp(Texto);
            Texto = strcat ('Eficiencia de productor = ', {' '}, num2str(EficienciasXFold(1,m)/folds), {' '}, 'IC = ', {' '}, num2str(std(EficienciasXFold(1,m)/folds)));
            disp(Texto);
            Texto = strcat ('Eficiencia de usuario = ', {' '}, num2str(EficienciasXFold(2,m)/folds), {' '}, 'IC = ', {' '}, num2str(std(EficienciasXFold(2,m)/folds)));
            disp(Texto);
        end
    end
    arbolesEstimado = mode(vec);
    Texto = strcat('Entre todas las iteraciones, la mejor eficiencia fue para la cantidad de árboles = ', {' '}, num2str(arbolesEstimado));
    disp(Texto);
end