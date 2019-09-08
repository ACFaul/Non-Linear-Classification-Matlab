load fisheriris;
% extract two attributes
pl = meas(:,3); % petal length
pw = meas(:,4); % petal width
figure;
h1 = gscatter(pl, pw, species,'rgb','os^');
legend('Setosa','Versicolor','Virginica','Location','best')
hold on;
X = [pl,pw];
cls = ClassificationDiscriminant.fit(X,species,...
    'DiscrimType','quadratic');
% plot the classification boundaries
% retrieve the coefficients for the quadratic boundary between 
% the first and second class (setosa and versicolor).
c = cls.Coeffs(1,2).Const;
l = cls.Coeffs(1,2).Linear;
q = cls.Coeffs(1,2).Quadratic;

% plot the curve c + [x1,x2]*l + [x1,x2]*q*[x1,x2]' = 0:
f = @(x1,x2) c + l(1)*x1 + l(2)*x2 + q(1,1)*x1.^2 + ...
    (q(1,2)+q(2,1))*x1.*x2 + q(2,2)*x2.^2;
h2 = ezplot(f,[.9 7.1 0 1]);
set(h2, 'Color','k');
% retrieve the coefficients for the quadratic boundary between 
% the second and third class (versicolor and viriginica).
c = cls.Coeffs(2,3).Const;
l = cls.Coeffs(2,3).Linear;
q = cls.Coeffs(2,3).Quadratic;
% plot the curve c + [x1,x2]*l + [x1,x2]*q*[x1,x2]' = 0:
f = @(x1,x2) c + l(1)*x1 + l(2)*x2 + q(1,1)*x1.^2 + ...
    (q(1,2)+q(2,1))*x1.*x2 + q(2,2)*x2.^2;
h3 = ezplot(f,[.5 7 0 2.5]);
set(h3, 'Color','k');
xlabel('Petal length');
ylabel('Petal width');
axis([0.8 7 0 3])
title('');
