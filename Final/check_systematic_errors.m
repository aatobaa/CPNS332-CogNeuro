numRuns = 100; 

systematicErrors = zeros(numRuns,8);
accuracy = zeros(numRuns,1);

for i = 1:numRuns
    i
    [pred, act,accur] = iterate_decoder();
    for dir = 1:8
        systematicErrors(i,dir) = (sum(pred==dir)-sum(act==dir))/length(act);
    end
    accuracy(i) = accur;
end
%%
figure
hold on
for z = 1:numRuns
    plot(1:8,systematicErrors(z,:),'.')
end
line([1 8], [0 0]);
plot(1:8,mean(systematicErrors,1),'LineWidth',2)
xlabel('Movement Direction')
ylabel('% of times overpredicted')
title('Systematic Errors')

figure
hist(accuracy)
