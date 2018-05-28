function [Yest, YestContinuo] = testSVM(Model, Xtest)
    [Yest, YestContinuo] = simlssvm(Model, Xtest);
end