function run_heat(K1, K2, K3)
    if any([K1 K2 K3] <= 0)
        error('Parameters must be positive')
    end
    [Q, dQdK] = heat_with_gradient(K1, K2, K3);
    writematrix(Q, 'qoi_value.dat', 'Delimiter', 'tab')
    writematrix(dQdK, 'qoi_jacobian.dat', 'Delimiter', 'tab')
end
