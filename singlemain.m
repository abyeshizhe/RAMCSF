clear 
close all
load dig1-10_uni

%% Initialization
maxIter = 50;
thresh = 1e-2;
c = length(unique(Y));                      % number of cluster
[n,n1] = size(X);                         % number of samples
m = 16;                                     % 
gamma = 1.4;                                % 1<gamma<2
M = [c+1,c+3,c+5,c+9,c+15,c+20,c+50,c+100]; % number of anchors
alpha = 1;
F = initialize(n,c);
G = rand(n1,c);
OBJ=[];
tic
%% Get Anchor Graph
[~,d] = size(X); 
[label, Anchors] = litekmeans(X, m); 
%B = double(ConstructA_NP(X', Anchors'));
B = X;

%% Optimization
for Iter = 1:maxIter
% Update F
for i = 1:n
    
        xVec = B(i,:);
    
    F(i,:) = searchBestIndicator(alpha, xVec, G, gamma);
end
obj = trace((B-F*G')'*(B-F*G'));
% Update G{v}
Ftemp = F*pinv(F'*F);
G = B'*Ftemp;



obj = trace((B-F*G')'*(B-F*G'));
%obj = sum((alpha^gamma)*W);


OBJ = [OBJ obj];

% obj(Iter) = 0;
% for v = 1: numView
%     obj(Iter) = obj(Iter) + (alpha(v)^gamma)*W{v};
% end
% if(Iter > 1)
%     diff = obj(Iter-1) - obj(Iter);
%     if(diff < thresh)
%         break;
%     end
% end
end
[maxv,ind]=max(F,[],2);
Result = ClusteringMeasure(Y, ind)
toc