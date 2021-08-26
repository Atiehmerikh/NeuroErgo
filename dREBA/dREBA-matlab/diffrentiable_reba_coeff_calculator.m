function [ws] =diffrentiable_reba_coeff_calculator(M,N,A)
%     M is a n to m matrix
%     N is a n to 1 matrix
%     A is a m to 3 matrix
%     n is number of postures
%     m is number of body degrees

    w = sym('w',[1 size(M,2)]);
    Q = M.^2 .* (A(:,1).') + M .* (A(:,2).') + (A(:,3).');
    disp('Q is computed')    
    
    ep = (sum(Q .* w, 2) - N).^2;
    disp('ep is calculated');
    
    df = arrayfun(@(x)(gradient(x, w) == 0), ep, 'UniformOutput', false);
    df = cellfun(@transpose,df,'un',0);
    df = vertcat(df{:});
    disp('df is ready');
    disp(size(df));

    sol = solve(df, w);
    disp('df is solved');
    
    ws = zeros(1,size(M,2));
    for i = 1:21
        ws(i) = eval(strcat('double(sol.w', num2str(i), ');'));
    end
    