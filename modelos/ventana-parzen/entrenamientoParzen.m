function Yesti = entrenamientoParzen(X,Xent,h)

%%% La funci�n debe retornar el valor de predicci�n Yesti para cada una de 
%%% las muestras en Xval. Por esa raz�n Yesti se inicializa como un vectores 
%%% de ceros, de dimensi�n M.
  
  M = size(X,1);
  N = size(Xent,1);
      
  Yesti=zeros(M,1);
      for j=1:M
        ker = repmat(X(j,:),N,1) - Xent;
        n = size(N,1);
        for i=1:N
            n(i,1) = gaussianKernel(norm(ker(i,:))/h);
        end
        Yesti(j,1) = sum(n)/N;
      end
end