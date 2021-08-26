function [A] =poly_coeff_calculator()
    A=zeros(21,3);

    %% 1- neck(flexion extension):
    no_intervals=3;
    squared_error = 0;
    interval_points = [-60,0,20,60];
    interval_reba_value = [2,1,2,2];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(1,:) = [sol.a_2,sol.a_1,sol.a_0];

    % % fprintf(fileID,'neck flexion');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    % %%2- neck side bending

    no_intervals=3;
    interval_points = [-54,0, 54,90];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(2,:) = [sol.a_2,sol.a_1,sol.a_0];

    % %%3- neck twist
    no_intervals=3;
    interval_points = [-60,0, 60,90];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(3,:) = [sol.a_2,sol.a_1,sol.a_0];
    % % fprintf(fileID,'neck side bending');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);


    %% trunk flexion:
    no_intervals = 4;
    interval_points = [-30,0,20,60,90];
    interval_reba_value = [3,1,2,4];
    squared_error =0;
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(4,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'trunk');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% trunk side bending
    no_intervals = 3;
    interval_points = [-40,0, 40,60];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(5,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'trunk side bending');

    %% trunk twist
    no_intervals = 3;
    interval_points = [-35,0, 35,50];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(6,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'trunk side bending');

    %% leg:
    no_intervals = 3;
    interval_points = [0,30,60,130];
    interval_reba_value = [1,1,2];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(7,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'leg');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% right upper arm:
    no_intervals = 4;
    interval_points = [-20, 0, 20, 45,90,180];
    interval_reba_value = [2,1,2,3,4];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(8,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'upper arm');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% right upper arm  side abduction
    no_intervals = 2;
    interval_points = [-90,0,140,200];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(9,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'upper arm abduction');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% right shoulder raised
    no_intervals = 2;
    interval_points = [0, 30,60];
    interval_reba_value = [0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(10,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'shoulder raised');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% left upper arm:
    no_intervals = 4;
    interval_points = [-20, 0, 20, 45,90,180];
    interval_reba_value = [2,1,2,3,4];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(11,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'upper arm');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% left upper arm  side abduction
    no_intervals = 2;
    interval_points = [-90,0,140,200];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(12,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'upper arm abduction');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% left shoulder raised
    no_intervals = 2;
    interval_points = [0, 30,60];
    interval_reba_value = [0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(13,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'shoulder raised');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);


    %% right lower arm:
    no_intervals = 3;
    interval_points = [0, 60, 100,120];
    interval_reba_value = [2,1,2];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(14,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'lower arm');
    %fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% left lower arm:
    no_intervals = 3;
    interval_points = [0, 60, 100,120];
    interval_reba_value = [2,1,2];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(15,:) = [sol.a_2,sol.a_1,sol.a_0];
    % fprintf(fileID,'lower arm');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);
    %% right wrist flexion:
    no_intervals = 3;
    interval_points = [-53,-15,15,53];
    interval_reba_value = [2,1,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(16,:) = [sol.a_2,sol.a_1,sol.a_0];
    % % fprintf(fileID,'wrist flexion');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% right wrist bent from midline
    no_intervals = 3;
    interval_points = [-40,0, 30,40];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(17,:) = [sol.a_2,sol.a_1,sol.a_0];
    % % fprintf(fileID,'wrist bent midline');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% right wrist twist

    no_intervals = 2;
    interval_points = [-90,0, 90,180];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(18,:) = [sol.a_2,sol.a_1,sol.a_0];


    %% left wrist flexion:
    no_intervals = 3;
    interval_points = [-53,-15,15,53];
    interval_reba_value = [2,1,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(19,:) = [sol.a_2,sol.a_1,sol.a_0];
    % % fprintf(fileID,'wrist flexion');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% left wrist bent from midline
    no_intervals = 3;
    interval_points = [-40,0, 30,40];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(20,:) = [sol.a_2,sol.a_1,sol.a_0];
    % % fprintf(fileID,'wrist bent midline');
    % fprintf(fileID,'%12.8f %12.8f %12.8f\r\n',A);

    %% left wrist twist

    no_intervals = 2;
    interval_points = [-90,0, 90,180];
    interval_reba_value = [1,0,1];
    for i = 1:1:no_intervals
        squared_error = squared_error + square_error_poly_coeff(interval_reba_value(i),interval_points(i),interval_points(i+1));
    end
    var = symvar(squared_error);

    df_a0 = diff(squared_error,var(1))==0;
    df_a1 = diff(squared_error,var(2))==0;
    df_a2 = diff(squared_error,var(3))==0;

    sol = solve([df_a0, df_a1, df_a2], [var(1) var(2) var(3)]);

    % the coefficient of polynomial function
    A(21,:) = [sol.a_2,sol.a_1,sol.a_0];
end