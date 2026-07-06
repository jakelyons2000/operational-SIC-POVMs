% Irreducible Representation of S_9 for partition (5,4)
clc; clear;

%% 1) Generate the 8x42x42 array T for transpositions (1,2) to (1,9)

% Step 1a: Generate all 42 Standard Young Tableaux (SYT) for shape (5,4)
% We represent a tableau by the row number (1 or 2) for each integer 1..9
T_row = [];
for i = 0:511
    % Convert to binary array of length 9 (1s and 2s)
    seq = dec2bin(i, 9) - '0' + 1; 
    
    % Check if there are exactly five 1s and four 2s
    if sum(seq == 1) == 5 && sum(seq == 2) == 4
        valid = true;
        % Verify standard tableau condition: 1s must never be outnumbered by 2s as we read
        for k = 1:9
            if sum(seq(1:k) == 2) > sum(seq(1:k) == 1)
                valid = false;
                break;
            end
        end
        if valid
            T_row = [T_row; seq];
        end
    end
end

% Get matrix coordinates (row, col) for each integer in each SYT
r = zeros(42, 9);
c = zeros(42, 9);
for i = 1:42
    c1 = 0; c2 = 0;
    for j = 1:9
        r(i,j) = T_row(i,j);
        if T_row(i,j) == 1
            c1 = c1 + 1;
            c(i,j) = c1;
        else
            c2 = c2 + 1;
            c(i,j) = c2;
        end
    end
end

% Step 1b: Compute adjacent transpositions s_k = (k, k+1) using YOR
S = zeros(8, 42, 42);
for k = 1:8
    for i = 1:42
        % Axial distance d_k = content(k+1) - content(k)
        dk = (c(i, k+1) - r(i, k+1)) - (c(i, k) - r(i, k));
        S(k, i, i) = 1 / dk;
        
        % Off-diagonal elements for swapped tableaus
        if abs(dk) > 1
            swapped_row = T_row(i, :);
            swapped_row([k, k+1]) = swapped_row([k+1, k]); % swap k and k+1
            
            % Find the index of the new tableau
            for j = 1:42
                if isequal(T_row(j, :), swapped_row)
                    idx = j;
                    break;
                end
            end
            S(k, i, idx) = sqrt(1 - 1/dk^2);
        end
    end
end

% Step 1c: Construct (1, k+1) from adjacent transpositions
T = zeros(8, 42, 42);
T(1,:,:) = S(1,:,:); % (12)
curr = squeeze(S(1,:,:));
for k = 2:8
    sk = squeeze(S(k,:,:));
    curr = sk * curr * sk; % (1, k+1) = (k, k+1) * (1, k) * (k, k+1)
    T(k,:,:) = curr;
end

disp('1) 8x42x42 array T created.');

%% 2) Find the matrix P1 of rank 14 and trace 14

% Create a cell array M where M{1} = I, and M{k} = (1, k)
M = cell(1, 9);
M{1} = eye(42);
for k = 1:8
    M{k+1} = squeeze(T(k, :, :));
end

% Calculate Jucys-Murphy element X9 = sum_{k=1}^8 (k, 9)
% Note: (k, 9) = (1, k) * (1, 9) * (1, k)
X9 = zeros(42, 42);
for k = 1:8
    X9 = X9 + M{k} * M{9} * M{k};
end

% In shape (5,4), '9' can be in box (1,5) -> content 4, or box (2,4) -> content 2.
% X9 has eigenvalues 4 and 2. We project onto the eigenspace of 4.
P1 = (X9 - 2 * eye(42)) / 2;

fprintf('2) Matrix P1 calculated. Rank: %d, Trace: %.1f\n', rank(P1), trace(P1));

%% 3) Define P2 to P9

P = cell(1, 9);
P{1} = P1;

% P2 = (19).P1.(19)
P{2} = M{9} * P1 * M{9};

% P_k = (1, k-1) . P2 . (1, k-1) for k=3...9
for k = 3:9
    P{k} = M{k-1} * P{2} * M{k-1};
end

disp('3) Matrices P2 to P9 defined.');

%% 4) Run Tests

disp(' ');
disp('--- 4) Running Tests ---');

% Check P1 = P1^*
err1 = norm(P{1} - P{1}');
fprintf('P1 = P1^* (Error norm): %e\n', err1);

% Check P4^2 = P4
err2 = norm(P{4} * P{4} - P{4});
fprintf('P4^2 = P4 (Error norm): %e\n', err2);

% Check sum of all Pi is 3*I
sumP = zeros(42, 42);
for i = 1:9
    sumP = sumP + P{i};
end
err3 = norm(sumP - 3 * eye(42));
fprintf('Sum(Pi) = 3I (Error norm): %e\n', err3);

% Check P2.P7.P2 = P2/4
err4 = norm(P{2} * P{7} * P{2} - P{2} / 4);
fprintf('P2*P7*P2 = P2/4 (Error norm): %e\n', err4);

%% 5) Determine dimensionality of the centralizer of P1,...,P9

% We find the dimension of the subspace of matrices X such that [P_i, X] = 0.
% We construct the combined linear operator L(X) = [P_i, X] using Kronecker products
% and find its null space via the eigenvalues of L' * L.

disp(' ');
disp('5) Calculating centralizer dimensionality...');
LL = zeros(42^2, 42^2);
I42 = eye(42);

for i = 1:9
    % Li represents the operator X -> P_i*X - X*P_i on flattened X
    Li = kron(I42, P{i}) - kron(P{i}.', I42);
    LL = LL + Li' * Li;
end

% Null space dimension corresponds to zero eigenvalues
eigenvalues = eig(LL);
centralizer_dim = sum(eigenvalues < 0.5);

fprintf('Dimensionality of the centralizer of P1,...,P9 is: %d\n', centralizer_dim);