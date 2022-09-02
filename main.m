clear all
close all

load WebKB
% load mnist4
%  load NUS

numView = length(X);
for i=1:numView
    X{i}=double(X{i});
    [N,m]=size(X{i});
    X{i}=mapstd(X{i});
end
y=double(Y);
%% Initialization
maxIter = 10;
numView = length(X);
c = length(unique(Y));                      % number of cluster
[n,~] = size(X{1});                         % number of samples
alpha = ones(numView,1)/numView;
M=c+7;                                    % number of anchors
gammalist = [1e2];              
lambdalist =[1e0];
deta0=1e0;
% par: m gamma lambda deta0
% WebKB:c+7 1e2 1e0 1e3
% NUS: c+7 le2 le0 le4
% Mnist4 c+100 le2 le0



OBJJJJ=[];
ParRecord=[];
for II=1:size(gammalist,2)
    gamma=gammalist(II);
    for III=1:size(lambdalist,2)
        lambda=lambdalist(III);
        OBJJ=[];RESULT=[];T=[];
       for JJ = 1:1
        tic
       %% Get Anchor Graph
       
        for v = 1:size(X,2)
            [~,d] = size(X{v}); 
            [label, Anchors] = litekmeans(X{v}, M);
            B{v} = ConstructA_NP(X{v}', Anchors');
        end

       %% 初始化F
       
        for v = 1: numView
            F{v}=orth(randn(M,c));
            G{v}=orth(randn(M,c));
        end

       %% 初始化高斯核带宽
        for v = 1:numView
            temp{v}=(B{v}-B{v}*F{v}*G{v}').^2;
            deta{v}=deta0*sqrt(sum(sum(temp{v})/(2*n)));
        end

       %% Optimization

        for Iter = 1:maxIter

            % Update W
            for v = 1:numView
                temp1{v}=(B{v}-B{v}*F{v}*G{v}').^2;
                temp2{v}=(sum(temp1{v},2))./(2*deta{v}^2);
                W{v}=1e-3*diag(exp(-temp2{v})./(deta{v}^2));
            end

            % Update G
            for v = 1:numView
                A{v}=B{v}'*W{v}*B{v}*F{v};
                [AA{v},~,CC{v}] = svd(A{v},'econ');
                G{v} = AA{v}*CC{v};        
            end

            % Update F{i} 
            for v = 1:numView
                U{v} = B{v}'*W{v}*B{v}-2*lambda*B{v}'*B{v};
                H{v} = B{v}'*W{v}*B{v}*G{v};
                QQ{v} = 2*U{v}*F{v}+2*H{v};
                [DD{v},~,EE{v}]=svd(QQ{v},'econ');
                F{v} = DD{v}*EE{v};

            end

            % Update \alpha
            for v = 1:numView
                s(v) = trace((B{v}-B{v}*F{v}*G{v}')'*W{v}*(B{v}-B{v}*F{v}*G{v}')+lambda*(B{v}'*B{v}-F{v}*F{v}')^2);

            end
            delta = max(s./(2*gamma));
            bb = delta*ones(1,numView)-s./(2*gamma);
            [alpha, val,p] = SimplexQP_ALM(eye(numView), -bb', 1e-2,1.05,1);


            %objective value
            OBJ = 0; LABEL = 0;
            for v = 1:numView
                obj(v) = alpha(v)*s(v)+gamma*alpha(v)^2;
                OBJ = OBJ+obj(v);
                LABEL = LABEL+alpha(v)*B{v}*F{v};
            end
            OBJJ=[OBJJ, OBJ];
        end
        [maxv,ind]=max(LABEL,[],2);
        Result = ClusteringMeasure(y, ind);
        t=toc;
        T=[T,t];
        Result=[Result,t];
        RESULT = [RESULT;Result];
       end
     record=[mean(RESULT(:,1)),std(RESULT(:,1));
        mean(RESULT(:,2)),std(RESULT(:,2));
        mean(RESULT(:,3)),std(RESULT(:,3));
        mean(RESULT(:,4)),std(RESULT(:,4));
        mean(RESULT(:,5)),std(RESULT(:,5));
        mean(RESULT(:,6)),std(RESULT(:,6));
        mean(RESULT(:,7)),std(RESULT(:,7));
        mean(RESULT(:,8)),std(RESULT(:,8));]
        record=record';
    tempRecord=[gamma lambda record(1,:)];
    ParRecord=[ParRecord;tempRecord];
    end
end

