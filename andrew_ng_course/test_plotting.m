functions;
test = (randn(1000,1000));
ftt = fft(test);
test_norm = meanNorm(test);
ftt_norm = meanNorm(real(ftt));

subplot(2,2,1);
imagesc(test)#, colorbar, colormap gray;
subplot(2,2,2);
imagesc(test_norm)#, colorbar, colormap gray;
subplot(2,2,3);
imagesc(real(ftt))#, colorbar, colormap gray;
subplot(2,2,4);
imagesc(ftt_norm)#, colorbar, colormap gray;
# tightfig; <- Matlab Command
