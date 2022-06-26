function dldX = relu_backward(X, dldY)
    % error('Implement this!');
    dldX = dldY.*(X>0);
end
