
%Input
input  = csvread('before.txt');
output = csvread('after.txt');
%N/2+1 of N channels exist after inplace fft
stage  = 8;
CHAN   = floor(stage/2+1);

figure(1);
plot(input);
figure(2);
plot(output);

frequency  = reshape(output,CHAN,length(output)/CHAN);

figure(3);
topv = max(max(frequency));
for ii=1:CHAN
    subplot(CHAN,1,ii);
    plot(frequency(ii,:)/topv);
    axis([1, length(frequency(ii,:)), 0, 1])
end
