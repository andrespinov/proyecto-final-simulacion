disp('Ingrese el número correspondiente al modelo que desea entrenar:');
disp('1. Ventana de Parzen');
disp('2. K-Vecinos');
disp('3. Redes Neuronales Artificiales');
disp('4. Random Forest');
disp('5. Máquinas de Soporte Vectorial');
punto = input('');

% Carga de las muestras de testeo
load('training.csv');
Xtraining = training(:,1:28);
Ytraining = training(:,29);
load('testing.csv');
Xtest = testing(:,1:28);
Ytest = testing(:,29);

% Extracción de la clase mayoritaria y reemplazo por el conjunto con
% submuestreo
Xtraining = Xtraining(Ytraining ~= 2,:);
Ytraining = Ytraining(Ytraining ~= 2,:);
load('C2-DSample.mat');
Xtraining = [Xtraining; newClass(:,1:28)];
Ytraining = [Ytraining; newClass(:,29)];

% Carga de las muestras artificiales
load('C1-SMOTE.mat');
Xartificial = newClass(:,1:28);
Yartificial = newClass(:,29);
load('C3-SMOTE.mat');
Xartificial = [Xartificial;newClass(:,1:28)];
Yartificial = [Yartificial;newClass(:,29)];
load('C4-SMOTE.mat');
Xartificial = [Xartificial;newClass(:,1:28)];
Yartificial = [Yartificial;newClass(:,29)];

% Unión de los conjuntos
X = [Xtraining; Xartificial];
Y = [Ytraining; Yartificial];
% X = Xtraining;
% Y = Ytraining;
N = size(X,1);

if punto == 1 % Ventana de Parzen
   ventanaParzen(X, Y, Xtest, Ytest);
elseif punto == 2 % K-Vecinos
    kvecinos(X, Y, Xtest, Ytest);
elseif punto == 3 % Redes Neuronales Artificiales
    redesneuronales(X, Y, Xtest, Ytest, N);
elseif punto == 4 % Random Forest
    randomforest(X, Y, Xtest, Ytest, N);
elseif punto == 5 % Máquinas de Soporte Vectorial
    maquinas(X, Y, Xtest, Ytest, N);
end
