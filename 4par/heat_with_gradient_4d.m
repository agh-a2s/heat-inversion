
function [C, dC] = heat_with_gradient_4d(K1, K2, K3, s)
    % Problem formulation
    a = @(K1, K2, K3) @(x, u, du, v, dv) const_on_region(x, K1, K2, K3) * dot(du, dv);
    psi  = @(x) (abs(x) < 1) * exp(x^2 / (x^2 - 1));
    dpsi = @(x) - 2 * x / (x^2 - 1)^2 * psi(x);
    f    = @(s) @(x) psi(norm(x) / s);  % forcing
    df   = @(s) @(x) - norm(x)/s^2 * dpsi(norm(x) / s);
    g    = @(x) 1;  % Neumann condition on edges not incident to (0,0)

    % location of x_i points
    measured_points = [0.5 0.5; -0.5 0.5; 0.5 -0.5; 0.2 0.2];

    % radius of ball (in maximum norm) around x_i used as region of measurement
    % ideally should be larger than the mesh element size, otherwise
    % the accuracy of measurement computations will suffer
    radius = 0.2;

    N = size(measured_points, 1);

    % B-spline knot vector (for each of 3 unit squares comprising the domain)
    p = 2;
    elems = 10;
    knot = simple_knot(elems, p);

    % number of quadrature points (max implemented: 5)
    k = p + 1;
    k_rhs = k; % used to compute measurements, may need to be larger
               % for small mesh and/or small measurement region

    % Setup
    p = degree_from_knot(knot);

    knot = repeat_knot(knot, p);
    points = linspace(-1, 1, max(knot) + 1);

    bx = basis1d(p, points, knot);
    by = basis1d(p, points, knot);

    % unnecessary DoFs - lower left quadrant
    nx = number_of_dofs(bx);
    ny = number_of_dofs(by);
    cx = floor(nx / 2);
    cy = floor(ny / 2);
    fixed_dofs = cartesian(0:cx, 0:cy);

    tic
    fprintf("Solving\n");
    M = assemble_matrix(a(K1, K2, K3), bx, by);
    M = dirichlet_bc_matrix(M, fixed_dofs, bx, by);
    [L, U, P, Q] = lu(M);
    invM = @(b) Q * (U \ (L \ (P*b)));
    u = solve_heat_with_matrix(invM, f(s), g, fixed_dofs, bx, by, k);
    toc

    tic
    C = zeros(N, 1);
    for i = 1:N
        Q = uniform_measure(measured_points(i,:), radius);
        C(i) = quantity(u, Q, bx, by, k_rhs);
        fprintf("C%d = %f\n", i, C(i));
    end
    toc

    % FD approximation
%     solution = @(k1, k2, k3, s) solve_heat(a(k1, k2, k3), f(s), g, fixed_dofs, bx, by, k_rhs);
%     h = 1e-3;
%     
%     dC = zeros(N, 4);
%     tic
%     for i = 1:N
%         Q = uniform_measure(measured_points(i,:), radius);
%     
%         C_next = quantity(solution(K1 + h, K2, K3, s), Q, bx, by, k_rhs);
%         C_prev = quantity(solution(K1 - h, K2, K3, s), Q, bx, by, k_rhs);
%         dC(i,1) = (C_next - C_prev) / (2*h);
%     
%         C_next = quantity(solution(K1, K2 + h, K3, s), Q, bx, by, k_rhs);
%         C_prev = quantity(solution(K1, K2 - h, K3, s), Q, bx, by, k_rhs);
%         dC(i,2) = (C_next - C_prev) / (2*h);
%     
%         C_next = quantity(solution(K1, K2, K3 + h, s), Q, bx, by, k_rhs);
%         C_prev = quantity(solution(K1, K2, K3 - h, s), Q, bx, by, k_rhs);
%         dC(i,3) = (C_next - C_prev) / (2*h);
%         
%         C_next = quantity(solution(K1, K2, K3, s + h), Q, bx, by, k_rhs);
%         C_prev = quantity(solution(K1, K2, K3, s - h), Q, bx, by, k_rhs);
%         dC(i,4) = (C_next - C_prev) / (2*h);
%     
%         fprintf("FD dC%d/dK = [%f, %f, %f, %f]\n", i, dC(i,:));
%     end
%     toc
%     FD = dC(:,:);

    % Adjoint state
    dC = zeros(N, 4);
    tic
    for i = 1:N
        Q = uniform_measure(measured_points(i,:), radius);
        lambda = - solve_heat_with_matrix(invM, Q, @(x) 0, fixed_dofs, bx, by, k_rhs);
        dC(i,1) = eval_form(a(1,0,0), u, lambda, bx, by);
        dC(i,2) = eval_form(a(0,1,0), u, lambda, bx, by);
        dC(i,3) = eval_form(a(0,0,1), u, lambda, bx, by);

        lambda_fun = make_it_function(lambda, bx, by);
        dfs = df(s);
        dC(i,4) = integrate2d(@(x) - dfs(x) * lambda_fun(x), bx, by, k_rhs);
        fprintf("AS dC%d/dK = [%f, %f, %f, %f]\n", i, dC(i,:));
    end
    toc
%     AS = dC(:,:);
%     diff = AS - FD;
%     fprintf("Difference: %g\n", max(abs(diff(:))));
end

function u = solve_heat(a, f, g, fixed_dofs, bx, by, k)
    M = assemble_matrix(a, bx, by);
    M = dirichlet_bc_matrix(M, fixed_dofs, bx, by);
    u = solve_heat_with_matrix(@(b) M \ b, f, g, fixed_dofs, bx, by, k);
end

function u = solve_heat_with_matrix(invM, f, g, fixed_dofs, bx, by, k)
    F = assemble_rhs(f, g, bx, by, k);
    F = dirichlet_bc_rhs(F, fixed_dofs, bx, by);

    nx = number_of_dofs(bx);
    ny = number_of_dofs(by);

    u = reshape(invM(F), nx, ny);
end

function M = assemble_matrix(a, bx, by)
    nx = number_of_dofs(bx);
    ny = number_of_dofs(by);
    n = nx * ny;
    k = max(bx.p, by.p) + 1;

    idx = @(dof) linear_index(dof, bx, by);

    % Assemble the matrix
    M = sparse(n, n);
    for e = elements(bx, by)
      J = jacobian2d(e, bx, by);
      for q = quad_data2d(e, k, bx, by)
        basis = basis_evaluator2d(q.x, bx, by);

        for i = dofs_on_element2d(e, bx, by)
          [v, dv] = basis(i);
          for j = dofs_on_element2d(e, bx, by)
            [u, du] = basis(j);
            M(idx(i), idx(j)) = M(idx(i), idx(j)) + a(q.x, u, du, v, dv) * q.w * J;
          end
        end
      end
    end
end

function F = assemble_rhs(f, g, bx, by, k)
    nx = number_of_dofs(bx);
    ny = number_of_dofs(by);
    n = nx * ny;

    idx = @(dof) linear_index(dof, bx, by);

    % Assemble the right-hand side
    F = zeros(n, 1);
    for e = elements(bx, by)
      J = jacobian2d(e, bx, by);
      for q = quad_data2d(e, k, bx, by)
        basis = basis_evaluator2d(q.x, bx, by);
        for i = dofs_on_element2d(e, bx, by)
          [v, dv] = basis(i);
          F(idx(i)) = F(idx(i)) + f(q.x) * v * q.w * J;
        end
      end

      % Boundary integrals
      sides = boundary_edges(e, bx, by);
      for edge = edge_data(e, sides, k, bx, by)
        J = edge.jacobian;
        for q = edge.quad_data
          basis = basis_evaluator2d(q.x, bx, by);
          for i = dofs_on_element2d(e, bx, by)
            v = basis(i);
            F(idx(i)) = F(idx(i)) + g(q.x) * v * q.w * J;
          end
        end
      end
    end
end

function f = const_on_region(x, a, b, c)
  if x(1) < 0
    f = a;
  elseif x(2) > 0
    f = b;
  else
    f = c;
  end
end

function w = uniform_measure(x, h)
    area = 4 * h^2;
    w = @(y) (max(abs(x - y)) < h) / area;
end

function q = quantity(u, weight, bx, by, k)
    u = make_it_function(u, bx, by);
    q = integrate2d(@(x) u(x) * weight(x), bx, by, k);
end

% Integrate function over the domain of specified B-spline basis
% Order quadratures is determined by orders of B-splines (p + 1).
% Quadrature points are generated using the mesh defined by the basis.
%
% f      - function accepting point as two-element vector
% bx, by - 1D bases
%
% returns - integral of f over product of domains of bx and by
function value = integrate2d(f, bx, by, k)
  value = 0;
  for e = elements(bx, by)
    J = jacobian2d(e, bx, by);
    for q = quad_data2d(e, k, bx, by)
      value = value + f(q.x) * q.w * J;
    end
  end
end

% Compute value and gradient of combination of 1D B-splines at point x
function [val, grad] = evaluate_with_grad2d(u, x, bx, by)
  sx = find_span(x(1), bx);
  sy = find_span(x(2), by);

  valsx = evaluate_bspline_basis_ders(sx, x(1), bx, 1);
  valsy = evaluate_bspline_basis_ders(sy, x(2), by, 1);

  offx = sx - bx.p;
  offy = sy - by.p;

  val = 0;
  grad = [0 0];
  for i = 0:bx.p
    for j = 0:by.p
      c = u(offx + i, offy + j);
      val     = val + c * valsx(i + 1, 1) * valsy(j + 1, 1);
      grad(1) = grad(1) + c * valsx(i + 1, 2) * valsy(j + 1, 1);
      grad(2) = grad(2) + c * valsx(i + 1, 1) * valsy(j + 1, 2);
    end
  end
end

function val = eval_form(a, f, g, bx, by)
  f = make_it_function(f, bx, by);
  g = make_it_function(g, bx, by);
  k = max([bx.p by.p]) + 1;
  val = integrate2d(@(x) eval_form_at(a, f, g, x), bx, by, k);
end

function val = eval_form_at(a, f, g, x)
  [u, du] = f(x);
  [v, dv] = g(x);
  val = a(x, u, du, v, dv);
end

% f - either function handle or 2D array of coefficients of tensor product B-spline basis
%
% Allows using the same code for both regular functions and B-splines in the norm functions.
function f = make_it_function(f, bx, by)
  if (~isa(f, 'function_handle'))
    f = @(x) evaluate_with_grad2d(f, x, bx, by);
  end
end

% Build cartesian product of specified vectors.
% Vector orientation is arbitrary.
%
% Order: first component changes fastest
%
% a1, a2, ... - sequence of n vectors
%
% returns - array of n-columns containing all the combinations of values in aj
function c = cartesian(varargin)
  n = nargin;

  [F{1:n}] = ndgrid(varargin{:});
  for i = n:-1:1
    c(i,:) = F{i}(:);
  end
end

% Create a row vector of size n filled with val
function r = row_of(val, n)
  r = val * ones(1, n);
end


% Index conventions
%------------------
%
% DoFs             - zero-based
% Elements         - zero-based
% Knot elements    - zero-based
% Linear indices   - one-based


% Create an one-dimensional basis object from specified data.
% Performs some simple input validation.
%
% For a standard, clamped B-spline basis first and last elements of the knot vector
% should be repeated (p+1) times.
%
% p       - polynomial order
% points  - increasing sequence of values defining the mesh
% knot    - knot vector containing integer indices of mesh points (starting from 0)
%
% returns - structure describing the basis
function b = basis1d(p, points, knot)
  assert(max(knot) == length(points) - 1, sprintf('Invalid knot index: %d, points: %d)', max(knot), length(points)));

  b.p = p;
  b.points = points;
  b.knot = knot;
end

% Number of basis functions (DoFs) in the 1D basis
function n = number_of_dofs(b)
  n = length(b.knot) - b.p - 1;
end

% Number of elements the domain is subdivided into
function n = number_of_elements(b)
  n = length(b.points) - 1;
end

% Domain point corresponding to i-th element of the knot vector
function x = knot_point(b, i)
  x = b.points(b.knot(i) + 1);
end

% Row vector containing indices of all the DoFs
function idx = dofs1d(b)
  n = number_of_dofs(b);
  idx = 0 : n-1;
end

% Enumerate degrees of freedom in a tensor product of 1D bases
%
% b1, b2, ...  - sequence of n 1D bases
%
% returns - array of indices (n-columns) of basis functions
function idx = dofs(varargin)
  if (nargin == 1)
    idx = dofs1d(varargin{:});
  else
    ranges = cellfun(@(b) dofs1d(b), varargin, 'UniformOutput', false);
    idx = cartesian(ranges{:});
  end
end

% Row vector containing indices of all the elements
function idx = elements1d(b)
  n = number_of_elements(b);
  idx = 0 : n-1;
end

% Enumerate element indices for a tensor product of 1D bases
%
% b1, b2, ...  - sequence of n 1D bases
%
% returns - array of indices (n-columns) of element indices
function idx = elements(varargin)
  if (nargin == 1)
    idx = elements1d(varargin{:});
  else
    ranges = cellfun(@(b) elements1d(b), varargin, 'UniformOutput', false);
    idx = cartesian(ranges{:});
  end
end

% Index of the first DoF that is non-zero over the specified element
function idx = first_dof_on_element(e, b)
  %idx = lookup(b.knot, e) - b.p - 1;
  [~, idx] = histc(e, b.knot);
  idx = idx - b.p - 1;
end

% Row vector containing indices of DoFs that are non-zero over the specified element
%
% e - element index (scalar)
% b - 1D basis
function idx = dofs_on_element1d(e, b)
  a = first_dof_on_element(e, b);
  idx = a : a + b.p;
end

% Row vector containing indices (columns) of DoFs that are non-zero over the specified element
%
% e      - element index (pair)
% bx, by - 1D bases
function idx = dofs_on_element2d(e, bx, by)
  rx = dofs_on_element1d(e(1), bx);
  ry = dofs_on_element1d(e(2), by);
  idx = cartesian(rx, ry);
end

% Determine which edges of the element lie on the domain boundary
%
% e      - element index (pair)
% bx, by - 1D bases
%
% returns - array of 4 boolean values (0 or 1), 1 meaning the edge is part of domain boundary
%           Order of the edges:
%             1 - left
%             2 - right
%             3 - top
%             4 - bottom
function s = boundary_edges(e, bx, by)
  nx = number_of_elements(bx);
  ny = number_of_elements(by);

  s = [e(1) == 0,       ...   % left
       e(1) == nx - 1,  ...   % right
       e(2) == ny - 1,  ...   % top
       e(2) == 0];            % bottom
end

% Compute 1-based, linear index of tensor product DoF.
% Column-major order - first index component changes fastest.
%
% dof           - n-tuple index
% b1, b2,, ...  - sequence of n 1D bases
%
% returns - linearized scalar index
function idx = linear_index(dof, varargin)
  n = length(varargin);

  idx = dof(n);
  for i = n-1 : -1 : 1
    ni = number_of_dofs(varargin{i});
    idx = dof(i) + idx * ni;
  end

  idx = idx + 1;
end

% Assuming clamped B-spline basis, compute the polynomial order based on the knot
function p = degree_from_knot(knot)
  p = find(knot > 0, 1) - 2;
end


% Spline evaluation functions are based on:
%
%    The NURBS Book, L. Piegl, W. Tiller, Springer 1995


% Find index i such that x lies between points corresponding to knot(i) and knot(i+1)
function span = find_span(x, b)
  low  = b.p + 1;
  high = number_of_dofs(b) + 1;

  if (x >= knot_point(b, high))
    span = high - 1;
  elseif (x <= knot_point(b, low))
    span = low;
  else
    span = floor((low + high) / 2);
    while (x < knot_point(b, span) || x >= knot_point(b, span + 1))
      if (x < knot_point(b, span))
        high = span;
      else
        low = span;
      end
      span = floor((low + high) / 2);
    end
  end
end

% Compute values at point x of (p+1) basis functions that are nonzero over the element
% corresponding to specified span.
%
% span  - span containing x, as computed by function find_span
% x     - point of evaluation
% b     - basis
%
% returns - vector of size (p+1)
function out = evaluate_bspline_basis(span, x, b)
  p = b.p;
  out = zeros(p + 1, 1);
  left = zeros(p, 1);
  right = zeros(p, 1);

  out(1) = 1;
  for j = 1:p
    left(j)  = x - knot_point(b, span + 1 - j);
    right(j) = knot_point(b, span + j) - x;
    saved = 0;

    for r = 1:j
      tmp = out(r) / (right(r) + left(j - r + 1));
      out(r) = saved + right(r) * tmp;
      saved = left(j - r + 1) * tmp;
    end
    out(j + 1) = saved;
  end
end

% Compute values and derivatives of order up to der at point x of (p+1) basis functions
% that are nonzero over the element corresponding to specified span.
%
% span  - span containing x, as computed by function find_span
% x     - point of evaluation
% b     - basis
%
% returns - array of size (p+1) x (der + 1) containing values and derivatives
function out = evaluate_bspline_basis_ders(span, x, b, der)
  p = b.p;
  out = zeros(p + 1, der + 1);
  left = zeros(p, 1);
  right = zeros(p, 1);
  ndu = zeros(p + 1, p + 1);
  a = zeros(2, p + 1);

  ndu(1, 1) = 1;
  for j = 1:p
    left(j)  = x - knot_point(b, span + 1 - j);
    right(j) = knot_point(b, span + j) - x;
    saved = 0;

    for r = 1:j
      ndu(j + 1, r) = right(r) + left(j - r + 1);
      tmp = ndu(r, j) / ndu(j + 1, r);
      ndu(r, j + 1) = saved + right(r) * tmp;
      saved = left(j - r + 1) * tmp;
    end
    ndu(j + 1, j + 1) = saved;
  end

  out(:, 1) = ndu(:, p + 1);

  for r = 0:p
    s1 = 1;
    s2 = 2;
    a(1, 1) = 1;

    for k = 1:der
      d = 0;
      rk = r - k;
      pk = p - k;
      if (r >= k)
        a(s2, 1) = a(s1, 1) / ndu(pk + 2, rk + 1);
        d = a(s2, 1) * ndu(rk + 1, pk + 1);
      end
      j1 = max(-rk, 1);
      if (r - 1 <= pk)
        j2 = k - 1;
      else
        j2 = p - r;
      end
      for j = j1:j2
        a(s2, j + 1) = (a(s1, j + 1) - a(s1, j)) / ndu(pk + 2, rk + j + 1);
        d = d + a(s2, j + 1) * ndu(rk + j + 1, pk + 1);
      end
      if (r <= pk)
        a(s2, k + 1) = -a(s1, k) / ndu(pk + 2, r + 1);
        d = d + a(s2, k + 1) * ndu(r + 1, pk + 1);
      end
      out(r + 1, k + 1) = d;
      t = s1;
      s1 = s2;
      s2 = t;
    end
  end

  r = p;
  for k = 1:der
    for j = 1:p+1
      out(j, k + 1) = out(j, k + 1) * r;
    end
    r = r * (p - k);
  end

end

% Evaluate combination of 2D B-splines at point x
function val = evaluate2d(u, x, bx, by)
  sx = find_span(x(1), bx);
  sy = find_span(x(2), by);

  valsx = evaluate_bspline_basis(sx, x(1), bx);
  valsy = evaluate_bspline_basis(sy, x(2), by);

  offx = sx - bx.p;
  offy = sy - by.p;

  val = 0;
  for i = 0:bx.p
    for j = 0:by.p
      val = val + u(offx + i, offy + j) * valsx(i + 1) * valsy(j + 1);
    end
  end
end

% Returns a structure containing information about 1D basis functions that can be non-zero at x,
% with the following fields:
%   offset - difference between global DoF numbers and indices into vals array
%   vals   - array of size (p+1) x (der + 1) containing values and derivatives of basis functions at x
function data = eval_local_basis(x, b, ders)
  span = find_span(x, b);
  first = span - b.p - 1;
  data.offset = first - 1;
  data.vals = evaluate_bspline_basis_ders(span, x, b, ders);
end

% Compute value and derivative of specified 1D basis function, given data computed
% by function eval_local_basis
function [v, dv] = eval_dof1d(dof, data, b)
  v = data.vals(dof - data.offset, 1);
  dv = data.vals(dof - data.offset, 2);
end

% Compute value and gradient of specified 2D basis function, given data computed
% by function eval_local_basis
function [v, dv] = eval_dof2d(dof, datax, datay, bx, by)
  [a, da] = eval_dof1d(dof(1), datax, bx);
  [b, db] = eval_dof1d(dof(2), datay, by);
  v = a * b;
  dv = [da * b, a * db];
end

% Creates a wrapper function that takes 2D basis function index as argument and returns
% its value and gradient
function f = basis_evaluator2d(x, bx, by, ders)
  datax = eval_local_basis(x(1), bx, 1);
  datay = eval_local_basis(x(2), by, 1);
  f = @(i) eval_dof2d(i, datax, datay, bx, by);
end


% Value of 1D element mapping jacobian (size of the element)
function a = jacobian1d(e, b)
  a = b.points(e + 2) - b.points(e + 1);
end

% Value of 2D element mapping jacobian (size of the element)
function a = jacobian2d(e, bx, by)
  a = jacobian1d(e(1), bx) * jacobian1d(e(2), by);
end

% Row vector of points of the k-point Gaussian quadrature on [a, b]
function xs = quad_points(a, b, k)
  % Affine mapping [-1, 1] -> [a, b]
  map = @(x) 0.5 * (a * (1 - x) + b * (x + 1));
  switch (k)
    case 1
      xs = [0];
    case 2
      xs = [-0.5773502691896257645, ...
             0.5773502691896257645];
    case 3
      xs = [-0.7745966692414833770, ...
             0,                     ...
             0.7745966692414833770];
    case 4
      xs = [-0.8611363115940525752, ...
            -0.3399810435848562648, ...
             0.3399810435848562648, ...
             0.8611363115940525752];
    case 5
      xs = [-0.9061798459386639928, ...
            -0.5384693101056830910, ...
             0,                     ...
             0.5384693101056830910, ...
             0.9061798459386639928];
  end
  xs = map(xs);
end

% Row vector of weights of the k-point Gaussian quadrature on [a, b]
function ws = quad_weights(k)
  switch (k)
    case 1
      ws = [2];
    case 2
      ws = [1, 1];
    case 3
      ws = [0.55555555555555555556, ...
            0.88888888888888888889, ...
            0.55555555555555555556];
    case 4
      ws = [0.34785484513745385737, ...
            0.65214515486254614263, ...
            0.65214515486254614263, ...
            0.34785484513745385737];
    case 5
      ws = [0.23692688505618908751, ...
            0.47862867049936646804, ...
            0.56888888888888888889, ...
            0.47862867049936646804, ...
            0.23692688505618908751];
  end
  % Gaussian quadrature is defined on [-1, 1], we use [0, 1]
  ws = ws / 2;
end


% Create array of structures containing quadrature data for integrating over 1D element
%
% e - element index
% k - quadrature order
% b - 1D basis
%
% returns - array of k structures with fields
%              x - point
%              w - weight
function qs = quad_data1d(e, k, b)
  xs = quad_points(b.points(e(1) + 1), b.points(e(1) + 2), k);
  ws = quad_weights(k);

  for i = 1:k
      qs(i).x = xs(i);
      qs(i).w = ws(i);
  end

end

% Create array of structures containing quadrature data for integrating over 2D element
%
% e      - element index (pair)
% k      - quadrature order
% bx, by - 1D bases
%
% returns - array of structures with fields
%              x - point
%              w - weight
function qs = quad_data2d(e, k, bx, by)
  xs = quad_points(bx.points(e(1) + 1), bx.points(e(1) + 2), k);
  ys = quad_points(by.points(e(2) + 1), by.points(e(2) + 2), k);
  ws = quad_weights(k);

  for i = 1:k
    for j = 1:k
      qs(i, j).x = [xs(i), ys(j)];
      qs(i, j).w = ws(i) * ws(j);
    end
  end
  qs = reshape(qs, 1, []);

end

% Compute quarature data for integrating on selected edges of the 2D element
%
% e      - index of the element
% sides  - array of 4 boolean values, used to determine which edges to prepare data for.
%          Order of the edges:
%            1 - left
%            2 - right
%            3 - top
%            4 - bottom
% k      - order of the quadrature
% bx, by - 1D bases
%
% returns - array of structures containing fields:
%             jacobian  - jacobian of the edge parameterization
%             normal    - unit vector perpendicular to the edge
%             quad_data - points and weights of 1D quadrature on the edge
function es = edge_data(e, sides, k, bx, by)
  % Empty structure array
  es = struct('jacobian', [], 'normal', [], 'quad_data', []);

  if (sides(1))
    es(end+1) = edge_data_left(e, k, bx, by);
  end
  if (sides(2))
    es(end+1) = edge_data_right(e, k, bx, by);
  end
  if (sides(3))
    es(end+1) = edge_data_top(e, k, bx, by);
  end
  if (sides(4))
    es(end+1) = edge_data_bottom(e, k, bx, by);
  end
end


% Auxiliary functions - computing quadrature data for each single edge

function edge = edge_data_left(e, k, bx, by)
  x1 = bx.points(e(1) + 1);

  edge.jacobian = jacobian1d(e(2), by);
  edge.normal = [-1 0];
  edge.quad_data = quad_data1d(e(2), k, by);
  for i = 1:k
    edge.quad_data(i).x = [x1, edge.quad_data(i).x];
  end
end

function edge = edge_data_right(e, k, bx, by)
  x2 = bx.points(e(1) + 2);

  edge.jacobian = jacobian1d(e(2), by);
  edge.normal = [1 0];
  edge.quad_data = quad_data1d(e(2), k, by);
  for i = 1:k
    edge.quad_data(i).x = [x2, edge.quad_data(i).x];
  end
end

function edge = edge_data_bottom(e, k, bx, by)
  y1 = by.points(e(2) + 1);

  edge.jacobian = jacobian1d(e(1), bx);
  edge.normal = [0 -1];
  edge.quad_data = quad_data1d(e(1), k, bx);
  for i = 1:k
    edge.quad_data(i).x = [edge.quad_data(i).x, y1];
  end
end

function edge = edge_data_top(e, k, bx, by)
  y2 = by.points(e(2) + 2);

  edge.jacobian = jacobian1d(e(1), bx);
  edge.normal = [0 1];
  edge.quad_data = quad_data1d(e(1), k, bx);
  for i = 1:k
    edge.quad_data(i).x = [edge.quad_data(i).x, y2];
  end
end

% Modify matrix and right-hand side to enforce uniform (zero) Dirichlet boundary conditions
%
% M      - matrix
% F      - right-hand side
% dofs   - degrees of freedom to be fixed
% bx, by - 1D bases
%
% returns - modified M and F
function [M, F] = dirichlet_bc_uniform(M, F, dofs, bx, by)
  for d = dofs
    i = linear_index(d, bx, by);
    M(i, :) = 0;
    M(i, i) = 1;
    F(i) = 0;
  end
end

function M = dirichlet_bc_matrix(M, dofs, bx, by)
  for d = dofs
    i = linear_index(d, bx, by);
    M(i, :) = 0;
    M(i, i) = 1;
  end
end

function F = dirichlet_bc_rhs(F, dofs, bx, by)
  for d = dofs
    i = linear_index(d, bx, by);
    F(i) = 0;
  end
end

% Evaluate function on a 2D cartesian product grid
%
% f      - function accepting 2D point as a two-element vector
% xs, ys - 1D arrays of coordinates
%
% returns - 2D array of values with (i, j) -> f( xs(j), ys(i) )
%           (this order is compatible with plotting functions)
function vals = evaluate_on_grid(f, xs, ys)
  [X, Y] = meshgrid(xs, ys);
  vals = arrayfun(@(x, y) f([x y]), X, Y);
end

% Subdivide xr and yr into N equal size elements
function [xs, ys] = make_grid(xr, yr, N)
  xs = linspace(xr(1), xr(2), N + 1);
  ys = linspace(yr(1), yr(2), N + 1);
end

% Plot 2D B-spline with coefficients u on a square given as product of xr and yr
%
% u      - matrix of coefficients
% xr, yr - intervals specifying the domain, given as two-element vectors
% N      - number of plot 'pixels' in each direction
% bx, by - 1D bases
%
% Domain given by xr and yr should be contained in the domain of the B-spline bases
function surface_plot_spline(u, xr, yr, N, bx, by)
  [xs, ys] = make_grid(xr, yr, N);
  vals = evaluate_on_grid(@(x) evaluate2d(u, x, bx, by), xs, ys);
  surface_plot_values(vals, xs, ys);
end

% Plot arbitrary function on a square given as product of xr and yr
%
% f      - function accepting 2D point as a two-element vector
% xr, yr - intervals specifying the domain, given as two-element vectors
% N      - number of plot 'pixels' in each direction
function surface_plot_fun(f, xr, yr, N)
  [xs, ys] = make_grid(xr, yr, N);
  vals = evaluate_on_grid(f, xs, ys);
  surface_plot_values(vals, xs, ys);
end

% Plot array of values
%
% vals   - 2D array of size [length(ys), length(xs)]
% xs, ys - 1D arrays of coordinates
function surface_plot_values(vals, xs, ys)
  surf(xs, ys, vals);
  xlabel('x');
  ylabel('y');
end

% Function pasting two copies of the knot vector together
function k = repeat_knot(knot, p)
  m = max(knot);
  k = [knot(1:end-1), knot(p+2:end) + m];
end

% Create a knot without interior repeated nodes
%
% elems - number of elements to subdivide domain into
% p     - polynomial degree
function knot = simple_knot(elems, p)
  pad = ones(1, p);
  knot = [0 * pad, 0:elems, elems * pad];
end

