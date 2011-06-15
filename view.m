
%Input
input    = csvread('before.txt');
raw_data = csvread('after.txt');
CHAN     = 8;

figure(1);
subplot(2,1,1);
plot(raw_data);
subplot(2,1,2);
tmp = abs(fftshift(fft(input,CHAN)));
plot(tmp(length(tmp)/2:end));

reshaped  = reshape(raw_data,CHAN,length(raw_data)/CHAN);
frequency = abs(fft(reshaped,[],1));


figure(2);
topv = max(max(frequency));
for ii=1:CHAN
    subplot(CHAN,1,ii);
    plot(frequency(ii,:)/topv);
    axis([1, length(frequency(ii,:)), 0, 1])
end

figure(3);
plot(rot90(reshaped));
