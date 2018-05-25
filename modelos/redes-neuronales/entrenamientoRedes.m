function net = entrenamientoRedes(Xent,Yent, capas_neuronas, ep )

    net = patternnet(capas_neuronas);
    net = configure(net,Xent',Yent');
    
    net.trainParam.epochs=ep;
    [net, tr]= train(net,Xent',Yent');
    plotperform(tr);

end

