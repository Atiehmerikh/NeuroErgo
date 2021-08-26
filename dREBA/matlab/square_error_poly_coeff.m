function [squared_error] =square_error_poly_coeff(reba_table,start_point,end_point)
    %This is going to calculate the integral calculation of squared error
    %of differentiable reba function and the one obtained from table for a specific joint angle  
    
    
    %  dREBA = a_2*q^2+a_1*q+a_0
    syms q a_2 a_1 a_0
    Q = a_2 * q^2 + a_1 * q + a_0;
    squared_error = int((Q - reba_table)^2,q,start_point,end_point);   
end

