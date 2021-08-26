%% compute coefficients based on the generated data

% M is body degrees, a 21 x 21 matrix
% N is total reba scores for each row of M, a 21 x 1 matrix
M = csvread('data/input/M.csv');
N = csvread('data/input/N.csv');

A=poly_coeff_calculator();
display('A is found');

ws = diffrentiable_reba_coeff_calculator(M, N,A);


%% test dREBA based on test data analogous to M and N, generated in Python

M_test = csvread('data/input/M_test.csv');
N_test = csvread('data/input/N_test.csv');


Q = M_test.^2 .* (A(:,1).') + M_test .* (A(:,2).') + (A(:,3).');
estimate = round(sum(Q .* ws, 2));

% adjust to the allowed range of REBA [1, 14]
estimate(estimate > 15) = 15;
estimate(estimate < 1) = 1;

% write test results
csvwrite('data/output/dREBA_estimate.csv', estimate);

ep = abs(estimate - N_test);
csvwrite('data/output/error_data.csv', ep);
