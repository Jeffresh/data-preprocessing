%% Obtención de datos
clear all; close all, clc
[X,y] = wine_dataset;
X = X';
y= y';

%% Distribución de las variables
boxplot(X)
title("Wine dataset: Distribución de valores por features")
xlabel('características') 
ylabel('Valores') 

%% Normalización
m = mean(X);
s = std(X);
normalized_X = (X - m)./s;
boxplot(normalized_X)
title("Wine dataset: Distribución de valores por features Normalizado")
xlabel('características') 
ylabel('Valores')
hold off;
%% Tratamiento de outliers
feats_outliers = [2, 3, 4, 5, 9, 10 ,11];
grade = 2;

m_outliers = normalized_X(:,feats_outliers);
limit(1,:) = mean(m_outliers) + std(m_outliers) * grade;
limit(2,:) = mean(m_outliers) - std(m_outliers) * grade;

new_x = normalized_X;

for i=1:length(feats_outliers)
    outliers = [find(normalized_X(:,feats_outliers(i)) > limit(1,i)); ...
        find(normalized_X(:,feats_outliers(i)) < limit(2,i))];
    new_x(outliers, feats_outliers(i)) = 0;
    
end
% 
% new_X = rmoutliers(normalized_X)
boxplot(new_x)
title("Wine dataset: Distribución de valores por features Normalizado")
xlabel('características') 
ylabel('Valores')
hold off;

%% Reducción de dimesionalidad PCA
new_y = zeros(length(y),1);

for i=1:size(y, 2)
    index = find(y(:, i)~=0);
    new_y(index, 1) = i;
end
new_y;

hold off;

[W, ~] = pca(new_x');

x_reduced = W * new_x';
x_reduced = x_reduced';

colors = 'rgb';

for i=1:max(new_y)
    ind = find(new_y==i);
    plot(x_reduced(ind,1), x_reduced(ind,2), strcat('x',colors(i))),hold on;
end
hold off;
title("Reducción de la dimesionalidad: PCA")
legend("Clase 1", "Clase 2", "Clase 3", "Location", "southwest")

%% Reducción de la dimensionalidad LDA

W = fisher(new_x', new_y', 2)
x_reduced = W * new_x';
x_reduced = x_reduced';

colors = 'rgb';

for i=1:max(new_y)
    ind = find(new_y==i);
    plot(x_reduced(ind,1), x_reduced(ind,2), strcat('x',colors(i))),hold on;
end
hold off;
title("Reducción de la dimesionalidad: LDA")
legend("Clase 1", "Clase 2", "Clase 3", "Location", "southwest")
