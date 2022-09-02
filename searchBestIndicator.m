function outVec = searchBestIndicator(alpha, xCell, G, gamma)
% solve the following problem,
numView = length(G);
c = size(G{1}, 2);
tmp = eye(c);
obj = zeros(c, 1);
for j = 1: c
    for v = 1: numView
        obj(j,1) = obj(j,1) + (alpha(v)^gamma) * (norm(xCell{v} - G{v}(:,j))^2);
    end
end
[~, min_idx] = min(obj);
outVec = tmp(:, min_idx);
end