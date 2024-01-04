clear;
load MatrixSolving.mat;
b = full(b);
if size(b,1) == 1
    b = b.';
end
tic;
phi = sparseK\b;
toc;

save phisolved.mat phi;