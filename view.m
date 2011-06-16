
%Input
input  = csvread('before.txt');
output = csvread('after.txt');

figure(1);
plot(input);

figure(2);
topv = max(max(output));
chns = size(output,2)
for ii=1:chns
    subplot(chns,1,ii);
    plot(output(:,ii)/topv);
    axis([1, length(output(:,ii)), 0, 1])
end
