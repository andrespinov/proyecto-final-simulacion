function redesneuronales(X,Y,Xtest,Ytest)
    
    porcentaje = 0.7;
    numero_clases = length(unique(Ytest));
    Errors = zeros(10,1);
    
    for k = 1:3

        Texto = strcat('Iteración k = ', {' '}, num2str(k));
        disp(Texto);
        
        % Se hace la partición entre los conjuntos de entrenamiento y validación.
        [Xtrain, Ytrain, Xval, Yval] = bootstrapping(X,Y, porcentaje);
        [Xtrain2, Ytrain2, Xval2, Yval2] = bootstrapping(Xtest,Ytest, porcentaje);        
        
        % Se llevan los datos al formato requerido por la red neuronal para
        % resolver problemas de clasificación multiclase
        Y_cod = zeros(size( Ytrain, 1), numero_clases);
        Y_cod2 = zeros(size( Yval, 1), numero_clases);
        Y_cod3 = zeros(size(Ytrain2, 1), numero_clases);
        Y_cod4 = zeros(size( Yval2, 1), numero_clases);
        
        for i = 1:numero_clases
            rows = Ytrain == i;
            rows2 = Yval == i;
            rows3 = Ytrain2 == i;
            rows4 = Yval2 == i;

            Y_cod( rows, i ) = 1;
            Y_cod2( rows2, i ) = 1;
            Y_cod3( rows3, i ) = 1;
            Y_cod4( rows4, i ) = 1;
        end
    
        Ytrain = Y_cod;
        Yval = Y_cod2;
        
        Ytrain2 = Y_cod3;
        Yval2 = Y_cod4;       

        %%% Normalización %%%
        [Xtrain,mu,sigma] = zscore(Xtrain);
        [Xval,mu,sigma] = zscore(Xval);       
        Xtrain2 = normalizar(Xtrain2, mu, sigma);
        Xval2 = normalizar(Xval2, mu, sigma);

        %%% Se crea y se entrena el modelo  %%%
        e = [10, 20, 30, 40, 50];
        c = [30, 34, 36, 38, 40];
        for i = 1: length(e)  
            Texto = strcat('Iteración i = ', {' '}, num2str(i));
            disp(Texto);
            for j = 1: length(c) 
                Texto = strcat('Iteración j = ', {' '}, num2str(j));
                disp(Texto);
                for h = 1: 10
                    model = entrenamientoRedes(Xtrain, Ytrain, c(j), e(i));
                    %%% Se aplica la regresión usando ANN-MLP  %%%
                    Yesti=validacionRedes(model,Xval);

                    %%% Se encuentra el error en la clasificación %%%
                    error = perform(model,Yval,Yesti');
                    Errors(h,1) = error;
                end 
                Texto = strcat('Eficiencia en [i,j] = ',{' '},num2str(1-mean(Errors)));
                disp(Texto);
                EficienciaI(i) = 1 - mean(Errors);
                EficienciaJ(j) = 1 - mean(Errors);
            end  
            %Se guarda el índice del j con mejor eficiencia
            [~,ind] = max(EficienciaI);
            vec(i) = ind;
            [~, ind] = max(EficienciaJ);
            vec2(j) = ind;
        end 
        eEstimadas = e(mode(vec));
        cEstimadas = c(mode(vec));        
        vec3(i) = eEstimadas;
        vec4(j) = cEstimadas;
        Texto = strcat('La mejor eficiencia fue para eEstimadas = ',{' '}, num2str(eEstimadas));
        disp(Texto);
        Texto = strcat('La mejor eficiencia fue para cEstimadas = ',{' '}, num2str(cEstimadas));
        disp(Texto);
        
        %Se repite el proceso para el conjunto de validación
        model = entrenamientoRedes(Xtrain2, Ytrain2, cEstimadas, eEstimadas);
        %%% Se aplica la regresión usando ANN-MLP  %%%
        Yesti=validacionRedes(model,Xval2);

        %%% Se encuentra el error en la clasificación %%%
        error = perform(model,Yval2,Yesti');
        Errors(h,1) = error;
        Eficiencia = 1- mean(Errors);
        Texto = strcat('Para el conjunto de testeo se obtuvo eficiencia = ',{' '},num2str(Eficiencia));
        disp(Texto);
    end  
end