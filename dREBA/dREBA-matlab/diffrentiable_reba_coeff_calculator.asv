function [ws] =diffrentiable_reba_coeff_calculator(M,N,A)
%     M is a n to m matrix
%     N is a n to 1 matrix
%     A is a m to 3 matrix
%     n is number of postures
%     m is number of body degrees
%       q = dlmread('body_angle_value_for_learning.txt',',');
%     a=dlmread('polynomial_coefficient.txt');
    w = sym('w',[1 size(M,2)]);
    %ep = sym(zeros(size(M,1),1));
    
    Q = M.^2 .* (A(:,1).') + M .* (A(:,2).') + (A(:,3).');
    disp('Q is computed')
    
%     for j=1:size(M,1)
%         Q(j,:) = (A(:,1).').*(M(j,:).^2) + (A(:,2).').*M(j,:) + (A(:,3).');
% %         for i=1:(size(M,2))
% %             Q(j,i) = A(i,1)*M(j,i)^2 + A(i,2)*M(j,i) + A(i,3);
% %         end
% %         Reba_table(j) = q(j,size(q,2));
%     end
    
    
    ep = (sum(Q .* w, 2) - N).^2;
%     for j=1:size(M,1)
%         for i=1:(size(M,2))
%             ep(j) =ep(j)+ w(i)*Q(j,i);
%         end
%         ep(j) = (ep(j)-N(j))^2;
%     end
    disp('ep is calculated');
    
    %df = zeros(size(M));
    df = arrayfun(@(x)(gradient(x, w) == 0), ep, 'UniformOutput', false);
    df = cellfun(@transpose,df,'un',0);
    df = vertcat(df{:});
%     for j=1:size(M,1)
%         df(j, :) = (gradient(ep(j), w) == 0);
% %         for i=1:(size(M,2))
% %             df(j,i) = diff(ep(j),w(i))==0;
% %         end
%     end
    disp('df is ready');
    disp(size(df));
    
    %sol = solve([df], [w]);
    sol = solve(df, w);
    disp('df is solved');
    
    ws = zeros(1,size(M,2));
    for i = 1:21
        ws(i) = eval(strcat('double(sol.w', num2str(i), ');'));
    end
    
%     w_1 = double(sol.w1);
%     w_2 = double(sol.w2);
%     w_3 = double(sol.w3);
%     w_4 = double(sol.w4);
%     w_5 = double(sol.w5);
%     w_6 = double(sol.w6);
%     
%     w_7 = double(sol.w7);
%     w_8 = double(sol.w8);
%     w_9 = double(sol.w9);
%     w_10 = double(sol.w10);
%     w_11 = double(sol.w11);
%     w_12 = double(sol.w12);
%     
%     w_13 = double(sol.w13);
%     w_14 = double(sol.w14);
%     w_15 = double(sol.w15);
%     w_16 = double(sol.w16);
%     w_17= double(sol.w17);
%     w_18 = double(sol.w18);
%     
%     w_19 = double(sol.w19);
%     w_20 = double(sol.w20);
%     w_21 = double(sol.w21);
    
    