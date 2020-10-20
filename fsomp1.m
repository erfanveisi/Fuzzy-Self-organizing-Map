clc
clear;
close all;

%% Data Preparation
load('sweet_bitter');
data = X';
y = zeros(max(max(class)), size(data,2));
for i = 1:size(data,2)
    y(class(i), i) = 1;
end

L=vec2ind(y);
Classes=unique(L)';
L=numel(Classes);



%% General Parameters
inputDim = size(data, 1);
outputDim = size(y, 1);
N = size(data, 2);
lattice = [10 10];        % Lattice size
ND = prod(lattice);     % Number of nodes

% SOM Parameters
maxItSOM = 40;          % Total SOM loops
sig = 2;                % Radius of net
eta = 0.5;              % Learning rate

% FSOM Parameters
maxItLVQ =12;          % Total LVQ2.1 loops
w0 = 5.5;               % Spreads definition constant
gv = 0.05;              % Learning rate of fuzzy sets
ga = 0.3;               % Learning rate of outputs
beta1 = 0.3;            % Maximum firing strength of the default rule
beta2 = 0.7;            % Default-Normal rule overlap ratio

%% ALGORITHM SECTION
% Initialization of fuzzy sets
sl = zeros(ND, inputDim);
c  = zeros(ND, inputDim);
sr = zeros(ND, inputDim);
for i = 1:ND        
    for j = 1:inputDim        
        c(i,j)  = randi([-10, 10]);
    end    
end

c1 = zeros(lattice);
c2 = zeros(lattice);
lr = lattice(1);
lc = lattice(1);
for ir = 1:lr
    for ic = 1:lc
        c1(ir,ic) = c(sub2ind(lattice, ir, ic), 1);
        c2(ir,ic) = c(sub2ind(lattice, ir, ic), 2);
    end
end
    
tic;
% FIRST STAGE OF LEARNING - UPDATE CENTERS
for t = 1:maxItSOM               
    
    % Calculate SOM learning params
    sigt = round(sig * (1-t/maxItSOM));
    etat = eta * (1-t/maxItSOM);
        
    for k = randperm(N)
        x = data(:,k);               
        
        % Calculate distance of sample to each neuron
        d = (x(1)-c1).^2 + (x(2)-c2).^2;    
        
        % Find the winner neuron        
        [mins, im] = min(d);
        [~,    mc] = min(mins);
        mr = im(mc);
        
        % Update the winner neuron
        c1(mr, mc) = c1(mr, mc) + etat * (x(1) -  c1(mr, mc));
        c2(mr, mc) = c2(mr, mc) + etat * (x(2) -  c2(mr, mc));
        
        % Update the neighbour neurons
        for s = 1:sigt
            ir = mr - s;
            ic = mc;
            if ir >= 1
                c1(ir, ic) = c1(ir, ic) + etat * (x(1) -  c1(ir, ic));
                c2(ir, ic) = c2(ir, ic) + etat * (x(2) -  c2(ir, ic));
            end
            
            ir = mr + s;
            ic = mc;
            if ir <= lr
                c1(ir, ic) = c1(ir, ic) + etat * (x(1) -  c1(ir, ic));
                c2(ir, ic) = c2(ir, ic) + etat * (x(2) -  c2(ir, ic));
            end
            
            ir = mr;
            ic = mc - s;
            if ic >= 1
                c1(ir, ic) = c1(ir, ic) + etat * (x(1) -  c1(ir, ic));
                c2(ir, ic) = c2(ir, ic) + etat * (x(2) -  c2(ir, ic));
            end
            
            ir = mr;
            ic = mc + s;
            if ic <= lc
                c1(ir, ic) = c1(ir, ic) + etat * (x(1) -  c1(ir, ic));
                c2(ir, ic) = c2(ir, ic) + etat * (x(2) -  c2(ir, ic));
            end
        end                              
    end          
end
toc

% Plot SOM result
figure
plot(data(1,:),data(2,:),'.b')
hold on
plot(c1,c2,'or')
plot(c1,c2,'k','linewidth',2)
plot(c1',c2','k','linewidth',2)
hold off
drawnow
axis equal



% SECOND STAGE OF LEARNING - FORMING FUZZY SETS AND RULE LABELING
for ir = 1:lr
    for ic = 1:lc
        c(sub2ind(lattice, ir, ic), 1) = c1(ir,ic);
        c(sub2ind(lattice, ir, ic), 2) = c2(ir,ic);
    end
end
for i = 1:ND            
    sl(i,:) = c(i,:) - w0;
    sr(i,:) = c(i,:) + w0;
end


alpha  = zeros(ND, N);
alpha0 = zeros(1,  N);
a0 = zeros(1, outputDim);
a  = zeros(ND, outputDim);
m  = zeros(inputDim, 1);

for k = 1:N
    x = data(:,k);     
    % Output of each rule
    for i = 1:ND
        for j = 1:inputDim                
            m(j) = tmf(x(j), sl(i,j), c(i,j), sr(i,j));
        end            
        [alpha(i, k), ind] = min(m);            
    end
    alphak = max(alpha(:, k));
    as = beta1 * (1-alphak/beta2);
    alpha0(k) = max([as 0]);
end


for k = 1:outputDim
    for i = 1:ND
        s = sum(alpha(i,:));
        if (s == 0)
            a(i, k) = 0;
        else
            a(i, k) = (alpha(i,:) * y(k,:)') / s; 
        end
    end
    
    s = sum(alpha0);
    if s == 0
        a0(k) = 0;
    else
        a0(k) = (alpha0 * y(k,:)') / s;
    end
end


% THIRD STAGE OF LEARNING - LVQ2.1
tic;
rmse = zeros(1, maxItLVQ);
for t = 1:maxItLVQ
    
    gv = 0.05 * (1-t/maxItLVQ);
    if ga > 0 
        ga = 0.1 * (1-t/6);
    end
    
    for k = 1:N
        x = data(:,k);     

        for i = 1:ND
            for j = 1:inputDim                
                m(j) = tmf(x(j), sl(i,j), c(i,j), sr(i,j));
            end            
            [alpha(i, k), ind] = min(m);         
        end
        alphak = max(alpha(:, k));
        as = beta1 * (1-alphak/beta2);
        alpha0(k) = max([as 0]);

        ys = zeros(outputDim, 1);
        % Compute output of FIS    
        for kk = 1:outputDim
            s = sum(alpha(:,k)) + alpha0(k);
            if s == 0
                ys(kk) = 0;
            else
                ss = 0;
                for i = 1:ND
                    ss = ss + alpha(i,k) .* a(i,kk);
                end
                ss = ss + alpha0(k) .* a0(kk);
                ys(kk) = ss / s;
            end
        end

        s = sum(alpha(:,k) > 0) + (alpha0(k) > 0);
        % At least two fuzzy rules fire simultaneously        
        if s > 1
            [~, ia] = sort([alpha0(k); alpha(:,k)], 'descend');
            iw = ia(1);
            ir = ia(2);

            % Winner and first runner-up are not default rule
            if iw ~= 1 && ir ~= 1
                iw = iw - 1;
                ir = ir - 1;
                                
                for j = 1:inputDim                
                    m(j) = abs(c(iw,j) - c(ir,j));
                end 
                [~, ik] = max(m);          
                
                if tmf(x(ik), sl(iw,ik), c(iw,ik), sr(iw,ik)) < tmf(x(ik), sl(ir,ik), c(ir,ik), sr(ir,ik))
                    tmp = iw;
                    iw = ir;
                    ir = tmp;
                end

                if c(ir,ik) < c(iw,ik)                            
                    if sign(y(:,k) - ys(:)) == sign(a(ir,:)' - a(iw,:)')
                        g = sr(ir,ik) + gv .* (c(iw,ik) - sr(ir,ik));
                    else
                        g = sr(ir,ik) + gv .* (sl(iw,ik) - sr(ir,ik));
                    end
                    if g > c(ir,ik)
                        sr(ir,ik) = g;
                    end
                else
                    if sign(y(:,k) - ys(:)) == sign(a(ir,:)' - a(iw,:)')
                        g = sl(ir,ik) + gv .* (c(iw,ik) - sl(ir,ik));
                    else
                        g = sl(ir,ik) + gv .* (sr(iw,ik) - sl(ir,ik));
                    end
                    if g < c(ir,ik)
                        sl(ir,ik) = g;
                    end
                end
            % Either winner or first runner-up is the default rule    
            else
                if ir == 1
                    ir = iw - 1;
                else
                    ir = ir - 1;
                end
                
                for j = 1:inputDim                
                    m(j) = tmf(x(j), sl(ir,j), c(ir,j), sr(ir,j));
                end 
                [~, ik] = min(m); 
                
                if x(ik) > c(ir,ik)
                    if sign(y(:,k) - ys(:)) == sign(a(ir,:)' - a0')
                        g = sr(ir,ik) - gv .* (sr(ir,ik) - c(ir,ik));
                    else
                        g = sr(ir,ik) + gv .* (sr(ir,ik) - c(ir,ik));
                    end
                    if g > c(ir,ik)
                        sr(ir,ik) = g;
                    end
                else
                    if sign(y(:,k) - ys(:)) == sign(a(ir,:)' - a0')
                        g = sl(ir,ik) - gv .* (c(ir,ik) - sr(ir,ik));
                    else
                        g = sl(ir,ik) + gv .* (c(ir,ik) - sr(ir,ik));
                    end
                    if g < c(ir,ik)
                        sl(ir,ik) = g;
                    end
                end
            end
            
        % Only one fuzzy rule fires
        elseif s == 1
            iw = find(alpha(:,k) > 0);
                        
            if iw ~= 1
                iw = iw - 1;
                
                % Update the centers
                for j = 1:inputDim 
                    c(iw,j) = c(iw,j) + gv .* (x(j) - c(iw,j));                
                end
                
                % Update the output of the winner rule
                a(iw,:) = a(iw,:) + ga .* alpha(iw,k) .* (y(:,k)' - ys');
            else
                % Update the output of the winner rule (default rule is
                % winner)
                a0 = a0 + ga .* alpha0(k) .* (y(:,k)' - ys');
            end
            
        end    
    end            
    
    %% Result
    oo = zeros(size(y));
    ooi = zeros(size(y));
    for k = 1:N
        x = data(:,k);     

        for i = 1:ND
            for j = 1:inputDim                
                m(j) = tmf(x(j), sl(i,j), c(i,j), sr(i,j));
            end            
            [alpha(i, k), ind] = min(m);         
        end
        alphak = max(alpha(:, k));
        as = beta1 * (1-alphak/beta2);
        alpha0(k) = max([as 0]);

        ys = zeros(outputDim, 1);
        % Compute output of FIS    
        for kk = 1:outputDim
            s = sum(alpha(:,k)) + alpha0(k);
            if s == 0
                ys(kk) = 0;
            else
                ss = 0;
                for i = 1:ND
                    ss = ss + alpha(i,k) .* a(i,kk);
                end
                ss = ss + alpha0(k) .* a0(kk);
                ys(kk) = ss / s;
            end
        end

        oo(:,k) = ys;
        [~, ik] = max(ys); 
        ooi(ik,k) = 1;        
    end
    rmse(t) = sqrt(norm(y - ooi))/N;
end
toc

%% Result
oo = zeros(size(y));
ooi = zeros(size(y));
for k = 1:N
    x = data(:,k);     
    
    for i = 1:ND
        for j = 1:inputDim                
            m(j) = tmf(x(j), sl(i,j), c(i,j), sr(i,j));
        end            
        [alpha(i, k), ind] = min(m);         
    end
    alphak = max(alpha(:, k));
    as = beta1 * (1-alphak/beta2);
    alpha0(k) = max([as 0]);

    ys = zeros(outputDim, 1);
    % Compute output of FIS    
    for kk = 1:outputDim
        s = sum(alpha(:,k)) + alpha0(k);
        if s == 0
            ys(kk) = 0;
        else
            ss = 0;
            for i = 1:ND
                ss = ss + alpha(i,k) .* a(i,kk);
            end
            ss = ss + alpha0(k) .* a0(kk);
            ys(kk) = ss / s;
        end
    end
    
    oo(:,k) = ys;
    [~, ik] = max(ys); 
    ooi(ik,k) = 1;
end

%% Plot Results

figure
plot(1:maxItLVQ, rmse);
title('RMSE');

disp('Number of misclassification: ');
disp(size(find(sum(y==ooi) ~= outputDim),2));




