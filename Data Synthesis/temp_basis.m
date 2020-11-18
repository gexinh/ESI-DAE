%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The script to obtain the Temporal Basis Function from the real data. 
% Input: X_real	
% Output: TBF matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data
load('./support_data/raw_data/channel.mat');
load('./support_data/raw_data/X_real.mat');
plot_flag = 0;
% -------------------------
%% remove the reference channels of real data
% ------------------------
c=channel.Channel;
dic=[]; %MAG dic
j=1;
for i=1:numel(c)
    if strcmp(c(i).Type,('MEG MAG'))
       dic(j)=i;
       j=j+1;
    end
end
clear c i j chan_sele
% --------------------------
%% parameter setting of real data
% --------------------------
sample=718;
fs=1793;
time=0.4; % -0.1 - 0.3 s
sti = -0.0998:1/fs:0.3001;
% 取0 到 0.15s作为源定位信号
min_=-100;
max_=250;%ms
time_point = find((0.001*min_)<=sti&sti<=(0.001*max_));  %270个点 180-448
pad = 2; %为了降采样补齐
sti= sti(1:time_point(end)+pad);

%% preprocess before obtaining the TBFs.
X = es.F(dic,1:(time_point(end)+pad));
% plot
if plot_flag == 1
figure()
hold on 
grid on 
xlabel('Time(ms)');
ylabel('Amplitude');
set(gca, 'Xlim', [min_ max_]);
plot(sti*1000,X');
end
% ------------------------
%% norm
% ------------------------
Normalize = 1;
% 
if Normalize
    ratio = max(abs(X(:)));
else
    ratio = 1;
end
X = X./ratio;

% parameters
srate = fs + 1; %HZ
drate = 9;    %降采样率
locutoff= 0 ;
% -----------------------------------------------------------------
% FIR filter : Reduce High-frequency component with a Lowpass filter
% -----------------------------------------------------------------
hicutoff = (1/2)* (srate/drate); %带宽要求3~4倍频，防止混叠 
epochframes = size(X,2); %450-18
[smoothdata,filtwts] = eegfilt(X,srate,locutoff,hicutoff,epochframes);
% plot 
%figure()
%plot(smoothdata');

% ------------------------
%       down-sample
% ------------------------
downX_ = zeros(4,epochframes/drate);
for i=1:size(X,1)
downX_(i,:)= downsample(smoothdata(i,:), drate);
end
if plot_flag == 1
 figure()
 plot(downX_');
end
% ------------------------
%% sefa0
% ------------------------
addpath('./support');

sti_ = -0.0998:drate/fs:0.3001;
time_point = find(10*0.001<=sti_&sti_<=250*0.001);
StimTime = find(abs(sti_-0)<1e-3);
sti_1= sti_(time_point(1):time_point(end));
sti_2 = sti_(1:time_point(end));
% StimTime=180; 
figure()
% different hyperparamets for sefa0

	Bpre=downX_(:,1:StimTime);
	comP = 10;
	[a,b,lam,alp,bet,Dic]=sefa0(downX_,Bpre,comP,3,100,0); % dic 就是求出的主成分 a是混淆矩阵
	alp = 1./diag(alp);
	[x,y] = sort(alp,'descend');  % 特征值和对应的序号
	alpha = alp(y);
	for i = 1:comP
		if sum(alpha(1:i))/sum(alpha) >= 0.90   %用来选取主成分的阈值？
			break;
		end

	end
Dic = Dic(y(1:i,:),:);
l2_n = sqrt(sum(Dic.^2,2));
Dic_ = Dic./repmat(l2_n,1,size(Dic,2));  %做一个范数normalization
TBF = Dic_(:,time_point(1):time_point(end));

%% plot ERP
if plot_flag == 1
	addpath('./support');
	num = 200;
	lamb = randfixedsum(4,num,1,0,1)';
	figure()
	hold on
	% set(gca, 'Xlim', [0 250]);
	for i = 1:num
		ERP= lamb(i,:)*TBF;
		plot(sti_1*1000, ERP);
	end
	title('Generate ERP signals')
end

%% save TBFs
TBF_ratio = max(abs(TBF(:))); 
TBF = TBF ./ TBF_ratio; % normalize to [-1, 1] 
save('./support_data/TBF_real.mat','TBF')
