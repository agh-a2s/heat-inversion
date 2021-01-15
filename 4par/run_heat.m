function run_heat_4par(K1, K2, K3, s)
    if any([K1 K2 K3 s] <= 0)
        error('Parameters must be positive')
    end
    
    psi  = @(x) (abs(x) < 1) * exp(x^2 / (x^2 - 1));
    dpsi = @(x) - 2 * x / (x^2 - 1)^2 * psi(x);
    
    % Source term (parametrized)
    f    = @(s) @(x) psi(norm(x) / s);
    % Source term derivative with respect to parameter(s)
    df   = @(s) @(x) - norm(x)/s^2 * dpsi(norm(x) / s);
    % Neumann condition on edges not incident to (0,0)
    g    = @(x) 1;
    % location of x_i points
    measured_points = [0.5 0.5; -0.5 0.5; 0.5 -0.5; 0.2 0.2];

    [Q, dQdK] = heat_with_gradient_4par(K1, K2, K3, s, f, df, g, measured_points);

    dlmwrite('qoi_value.dat', Q, 'delimiter', '\t', 'precision', 15)
    dlmwrite('qoi_jacobian.dat', dQdK, 'delimiter', '\t', 'precision', 15)
end
