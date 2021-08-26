clc;


M = csvread('data/M.csv');
N = csvread('data/N.csv');

display(size(M));
display(size(N));

A=poly_coeff_calculator();

display('A is found');

head = 21;
M = M(1:head, :);
N = N(1:head);
ws = diffrentiable_reba_coeff_calculator(M, N,A);

M_test = csvread('data/M_test_2.csv');
N_test = csvread('data/N_test_2.csv');


Q = M_test.^2 .* (A(:,1).') + M_test .* (A(:,2).') + (A(:,3).');
estimate = round(sum(Q .* ws, 2));
estimate(estimate > 15) = 15;
estimate(estimate < 1) = 1;

csvwrite('dREBA_estimate.csv', estimate);

ep = abs(estimate - N_test);

mean(ep)

csvwrite('error_data_3.csv', ep);
