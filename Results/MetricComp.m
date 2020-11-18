%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate the metric indicators for different algorithm 
% and image them as Box-plot.
% Author: Gexin Huang 
% Date: 2020/3/21
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Vert = 1024;
Case_num = 4;
fig_path = './result_fig';              %图片存放路径，手动设置
peak_t=17;
idc = {'ori','re','mne','lor','sbl','foc', 'stt'};
weight_path = '/DAE_TBF';       %总的路径 
plot_flag = 1;

% weight_path = '/1x1_new_conv';          
% weight_path = '/v3.15_1s_rand';
%% 单源
% weight_path = '/v3.15_1s_rand';   %保存图片的路径
% weight_path = '/v3.15_1s_rand_m';   %保存图片的路径

% data_path = '/v3.15_1s_rand';       %读取数据的路径
data_path = '/TBF_1s_';       %读取数据的路径
% weight_path = '/DAE_TBF_1s';   %保存metric的路径
%% 双源 
% weight_path = '/v3.15_2s_rand_m' ;     %保存图片的路径
% data_path = '/v3.15_2s_rand';       %读取数据的路径
% weight_path = '/DAE_TBF_2s';   %保存metric的路径
% data_path = '/TBF_2s_';       %读取数据的路径
%% path 
path = './data_for_test';
% path = [path, '/v_',int2str(Vert)];
% path = [path, '/v_',int2str(Vert),'_modify'];
path = [path, '/v_',int2str(Vert),'_TBF'];  % mixed
% path = [path, '/v_',int2str(Vert),'_fixed_TBF'];% fixed

metric_path = [path,'/metric_mat'];
metric_path = [metric_path,weight_path];
if ~exist(metric_path)
    mkdir(metric_path);
end


%% hyper parameters
lambda_mne=1.5;
lambda_loreta=3.5;

lambda_sbl = 0.25 ;  %经典算法的约束权重     %权重小一些，迭代次数可以延长试试 / 权重大，迭代次数少
lambda_foc = 0.35 ;                      
iters_sbl = 20;                           %40   % 0.35 20 
iters_foc = 35; %SBL FOCUSS的迭代次数     %40    % 0.3  40
%% load data
addpath('./metric');     %距离算法
addpath('./metric/generate_support');
addpath('./algorithm')  %经典源成像算法
% load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/Gain_l2_',int2str(Vert),'.mat']);
load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/Gain_n_',int2str(Vert),'.mat']);
load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/Cortex_',int2str(Vert),'.mat']);
load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/laplace_',int2str(Vert),'.mat']);
% load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/laplace_energy_',int2str(Vert),'.mat']);
adajacent_matrix = Cortex.VertConn;
VertCoor = Cortex.Vertices;    %
laplace_matrix = delta;            %
% laplace_energy_matrix = W;            %
leadfield = Gain_matrix;       %
count = 0;
%% preprocess
fig_path = [fig_path,weight_path];
if ~exist(fig_path)
    mkdir(fig_path);
end
%% 加入brian_list ，用以画脑图
brain_list = zeros(numel(idc),Vert,12); % [method]
%% 加入metric list ，用以画折线图
Mean_list = cell(Case_num,1);  % case_num 
Std_list = cell(Case_num,1);
% 每个list内存放一个4x3的矩阵
Mean_case_list = zeros(4,numel(idc)-1,4); % mode_num * algorithm_num * metric_num
Std_case_list =  zeros(4,numel(idc)-1,4);
% total list
Case_list = cell(Case_num,1);
flag = 0;
training_flag = 1;
%%
% for i = 1:Case_num
for c = 1:Case_num   %  
    path1 = [path ,'/case_',int2str(c)];
    %% case setting
    if c ==1
    %    noise_table = [0,3,5,10];
        noise_table = [-5,0,5,10];
        mode = 4;
    %     weight_path = '/DAE_TBF_1s';   %保存图片的路径
        data_path = '/TBF_1s_';       %读取数据的路径
        Mode_list = cell(mode ,1);
    elseif c==2
    %    area_table = [3,6,9,12];
       area_table = [3,6,10,15];
    %    noise_table = [5,5,5,5];
       noise_table = [-5,-5,-5,-5];
       mode = 4;
    %    weight_path = '/DAE_TBF_1s';   %保存图片的路径
       data_path = '/TBF_1s_';       %读取数据的路径
       Mode_list = cell(mode ,1);
    elseif c==3
        area_table = [4,4,;
                      4,10;
                      10,4;
                      10,10];  
        noise_table = [5,5,5,5];
        mode = 4;
    %     weight_path = '/DAE_TBF_2s';   %保存图片的路径
%         data_path = '/TBF_2s_';       %读取数据的路径
        data_path = '/TBFm_2s_';       %读取数据的路径
        Mode_list = cell(mode ,1);
    elseif c==4
        area_table = [
                     7, 7;
                     7, 7;
                     7, 7;
                      7, 7];
        noise_table = [5,5,5,5];
        corr_table = [0,0.3,0.6,0.9];
        mode = 4;
    %     weight_path = '/DAE_TBF_2s';   %保存图片的路径
%         data_path = '/TBF_2s_';       %读取数据的路径
        data_path = '/TBFm_2s_';       %读取数据的路径
        Mode_list = cell(mode ,1);
    end
    for j = 1:mode
        count = count + 1;
        if c == 1
            path2 = [path1,'/scalp_',int2str(noise_table(j)),'dB'];
        elseif c == 2
            path2 = [path1,'/source_area_',int2str(area_table(j))];
        elseif c == 3
            path2 = [path1,'/source_area_[',int2str(area_table(j,1)),', ',int2str(area_table(j,2)),']'];
        elseif c == 4
            path2 = [path1,'/erp_corr_',num2str(corr_table(j))];   
        end
        %% load my algorithm data
        data_path1=[data_path,int2str(noise_table(j)),'dB'];
        path3 = [path2, data_path1];

        s_ori=load([path2,'/source_t.mat']);
        s_re = load([path3,'/source_re.mat']);
        x_re = load([path3,'/scalp_re.mat']);
        x = load([path3,'/scalp.mat']);
        s_ori = abs(s_ori.gen_te);     %都选取绝对值进行比较
        s_ori = s_ori(:,:,2:end);   %变成40个时刻
        s_re = double(abs(s_re.source_re));  %都选取绝对值进行比较
        x = x.scalp;
        x_re = x_re.scalp_re;
        %% different method 
    %     s_mne = wMNE(x,leadfield,lambda_mne); %切分成100次来做 
    %     save([path3,'/s_mne.mat'],'s_mne')
    %     s_lor = LORETA(x,leadfield,laplace_matrix,lambda_loreta);
    %     save([path3,'/s_lor.mat'],'s_lor')
    %     s_sbl = SBL(x,leadfield,lambda_sbl,iters_sbl);
    %     save([path3,'/s_sbl.mat'],'s_sbl')
    %     s_foc = FOCUSS(x,leadfield,lambda_foc,iters_foc);
    %     save([path3,'/s_foc.mat'],'s_foc')
        %% wmne
        if ~exist([path3,'/s_mne.mat'])
            s_mne = MNE(x,leadfield,lambda_mne); %切分成100次来做
            save([path3,'/s_mne.mat'],'s_mne')
            s_mne = abs(s_mne);
        else
            load([path3,'/s_mne.mat'])
        end
        %% lor
        if ~exist([path3,'/s_lor.mat'])
            s_lor = LORETA(x,leadfield,Cortex,lambda_loreta);
            save([path3,'/s_lor.mat'],'s_lor')
            s_lor = abs(s_lor);
        else
            load([path3,'/s_lor.mat'])
        end
        %% sbl
        if ~exist([path3,'/s_sbl.mat'])
            s_sbl = SBL(x,leadfield,lambda_sbl,iters_sbl);
            save([path3,'/s_sbl.mat'],'s_sbl')
            s_sbl = abs(s_sbl);
        else
            load([path3,'/s_sbl.mat'])
        end
        %% foc
        if ~exist([path3,'/s_foc.mat'])
            s_foc = FOCUSS(x,leadfield,lambda_foc,iters_foc);
            save([path3,'/s_foc.mat'],'s_foc')
            s_foc = abs(s_foc);
        else
            load([path3,'/s_foc.mat'])
        end
        %% stt
        if ~exist([path3,'/s_stt.mat'])
            print( 'please use script to generate stt')
        else
            load([path3,'/s_stt.mat'])
            s_stt = abs(s_stt);  %计算一下绝对值
        end

        %% 再把每个数据的峰值时刻取出来
        if c ==4
            switch j
                case j==1
                    peak_t=31;
                case j==2
                    peak_t=24;
                case j==3
                    peak_t=23;
                case j==4
                    peak_t=18;
            end
        end
        for m = 1:numel(idc)
            eval(['s_',idc{m},'_set = s_',idc{m},'(1:end-1,:,peak_t);']);  % s_ori_set etc.
        end
        mtd = numel(idc)-1;%method的个数
        
        %% 计算Metric
        Metric = {'AUC','RMSE','SD','DLE'}; 
        if ~exist([metric_path,'/Mean_list.mat']) | ~exist([metric_path,'/Std_list.mat']) | ~exist([metric_path,'/Case_list.mat']) | training_flag == 1
            %% 生成列表
            
            for m = 1:numel(Metric)
                eval([Metric{m},'_list = zeros(size(s_ori_set,1),mtd);'])  % DLE_list etc.
            end
%             DLE_list = zeros(size(s_ori_set,1),mtd);
%             SD_list = zeros(size(s_ori_set,1),mtd);
%             RMSE_list = zeros(size(s_ori_set,1),mtd);
%             AUC_list = zeros(size(s_ori_set,1),mtd);
            Metric_list = cell(numel(Metric),1);
            for n = 1:mtd  %不同的算法
%                 eval(['DLE_Vec = DLE(s_',idc{1},'_set',', s_',idc{n+1},'_set',', VertCoor);']);
                eval(['DLE_Vec = utimate_DLE(s_',idc{1},'(1:end-1,:,:)',',s_',idc{n+1},'(1:end-1,:,:)',', VertCoor);']);
                DLE_list(:,n) = DLE_Vec;
                eval(['SD_Vec = SD(s_',idc{1},'_set',', s_',idc{n+1},'_set',',VertCoor);']);
                SD_list(:,n) = SD_Vec;
        %         eval(['RMSE_Vec =
        %         RMSE(s_',idc{1},'_set',',s_',idc{n+1},'_set',');']);  %只比较一个时刻
                eval(['RMSE_Vec = RMSE(s_',idc{1},'(1:end-1,:,:)',',s_',idc{n+1},'(1:end-1,:,:)',');']);
                RMSE_list(:,n) = RMSE_Vec;
                eval(['AUC_Vec = AUC(s_',idc{1},'(1:end-1,:,:)',',s_',idc{n+1},'(1:end-1,:,:)',', adajacent_matrix);']);
%                 eval(['AUC_Vec = AUC(s_',idc{1},'_set',',s_',idc{n+1},'_set',', adajacent_matrix);']);
                AUC_list(:,n) = AUC_Vec;
                % 排放顺序：AUC; RMSE; SD; DLE
%                 Mean_case_list(j,n,1) = mean(AUC_Vec);
%                 Std_case_list(j,n,1) = std(AUC_Vec);
%                 Mean_case_list(j,n,2) = mean(RMSE_Vec);
%                 Std_case_list(j,n,2) = std(RMSE_Vec);
%                 Mean_case_list(j,n,3) = mean(SD_Vec);
%                 Std_case_list(j,n,3) = std(SD_Vec);
%                 Mean_case_list(j,n,4) = mean(DLE_Vec);
%                 Std_case_list(j,n,4) = std(DLE_Vec);
                
                for m = 1:numel(Metric)
                     eval(['Mean_case_list','(j, n,',int2str(m),') = mean(',Metric{m},'_Vec);'])
                     eval(['Std_case_list','(j, n,',int2str(m),') = std(',Metric{m},'_Vec);'])
                end
                
            end 
        % 得到了metric_list
            for m = 1:numel(Metric)
                eval(['Metric_list{',int2str(m),',1} = ',Metric{m},'_list;']);
            end
            Mode_list{j,1} = Metric_list;  
        else
            flag = 1;
            
        end       
        
        %% 画箱型图boxplot
        Metric_title = {'AUC','RMSE','SD(mm)','DLE(mm)'};
    %     for m = 1:3
    %         figure(3*(count-1)+m)
    %         boxplot(eval([Metric{m},'_list;']),'Labels',{'My_aLg','MNE','LORETA'},'Whisker',2);
    %         xlabel('differet Methods')
    %         ylabel(Metric{m})
    %         title(['Case_',int2str(c),'Mode_',int2str(j),' of ',Metric{m}])
    %     end
           %% figure 1
%          fig = figure(count);
%         for m = 1:numel(Metric)  % metric
%             subplot(2,2,m)
%         boxplot(eval([Metric{m},'_list;']),...
%                 'Labels',{'Ours','wMNE','LORETA','SBL','FOCUSS','STTONNICA'},...
%                 'Whisker',1.5);
%     %       xlabel('differet Methods')
%             ylabel(Metric_title{m},'FontSize',8)
%             title(['Case_',int2str(c),'Mode_',int2str(j),' of ',Metric{m}],'FontSize',8)
%         end
        %% figure 2
%         fig = figure(count);
%         set(gcf,'position',[150 150 1000 200])
%         set(gca,'Position',[.15 .15 .8 .75]);  %从
%         for m = 1:numel(Metric)  % metric
%             subplot(1,4,m)
%             
%             if flag == 1
%                load([metric_path,'/Case_list.mat']);
%                boxplot(Case_list{c,1}{j,1}{m,1},...
%                     'Labels',{'Ours','wMNE','LOR','SBL','FOC','STTO'},...
%                     'Whisker',1.5);
%                 
%             else               
%             boxplot(eval([Metric{m},'_list;']),...
%                     'Labels',{'Ours','wMNE','LORETA','SBL','FOCUSS','STTONNICA'},...
%                     'Whisker',1.5);
%             end
%     %       xlabel('differet Methods')
%             ylabel(Metric_title{m},'FontSize',8)
%     %         title(['Case_',int2str(c),'Mode_',int2str(j),' of ',Metric{m},'FontSize',8])
%         end
    %     suptitle(['Case_',int2str(c),' Mode_',int2str(j)])
    %     saveas(fig,[fig_path,'/fig_',int2str(count),'.png'])   

     %% %%%%%%%%%%%%%%%%%%%%%%%% 分图 %%%%%%%%%%%%%%%%%%%%%%%%%        
    %     for m = 1:numel(Metric)  % metric
    %         fig = figure( (count-1)*numel(Metric) + m );
    % %         set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.9, 0.9]);
    %         figure_FontSize=8;
    %         boxplot(eval([Metric{m},'_list;']),...
    %             'Labels',{'My_aLg','wMNE','LORETA','SBL','FOCUSS'},...
    %             'Whisker',1.5);
    % %       xlabel('differet Methods')
    %         ylabel(Metric_title{m},'FontSize',figure_FontSize)
    % %         title(['Case_',int2str(c),'Mode_',int2str(j),' of ',Metric{m}])
    % %         saveas(fig,[fig_path,'/fig_',int2str((count-1)*numel(Metric) + m ),'.png'])
    % %         saveas(fig,[fig_path,'/fig_',int2str((count-1)*numel(Metric) + m ),'.eps'])
    %     end

    %     suptitle(['Case_',int2str(c),' Mode_',int2str(j)])

        %% 切分数据，把最后一个样本单独取出来
        for m = 1: numel(idc)
           eval(['s_',idc{m},'1 = squeeze(s_',idc{m},'(end,:,peak_t));']); % s_ori1 etc.
           brain_list(m,:,count) =eval(['s_',idc{m},'1'])';
        end
    end
    if flag == 0
        Mean_list{c,1} = Mean_case_list;
        Std_list{c,1} = Std_case_list;
        Case_list{c,1} =  Mode_list;
    end
    
end
%% plot all image 
if plot_flag == 1
c = 3;
% fig = figure();
% set(gcf,'position',[150 150 600 600])
% set(gca,'Position',[.05 .05 .95 .95]);  %从
for j = 1:mode     
        for m = 1:(numel(Metric)-1)  % metric
            fig = figure( (j-1)*(numel(Metric)-1) + m );
%             set(gcf,'position',[150 150 600 600])
%             set(gca,'Position',[.015 .015 .95 .95]);  %从
%             subplot(4,3,(j-1)*3+m)
%             set(gca,'Position',[.05 .05 .95 .95]);  %从
            load([metric_path,'/Case_list.mat']);
            boxplot(Case_list{c,1}{j,1}{m+1,1},...
                    'Labels',{'Ours','wMNE','LORETA','SBL','FOCUSS','STTONNICA'},...
                    'Whisker',1.5);
                
%             saveas(fig,['./result/metric/boxplot','/fig_',int2str((j-1)*(numel(Metric)-1) + m ),'.eps'])
            
    %       xlabel('differet Methods')
%             ylabel(Metric_title{m+1},'FontSize',8)
    %         title(['Case_',int2str(c),'Mode_',int2str(j),' of ',Metric{m},'FontSize',8])
        end
end
end

%% save data :先不考虑不同情况，只是单纯画出来
if flag == 0
save([metric_path,'/Mean_list.mat'],'Mean_list');
save([metric_path,'/Std_list.mat'],'Std_list');
save([metric_path,'/Case_list.mat'],'Case_list');
% save([metric_path,'/brain_list.mat'],'brain_list');
end

rmpath('./metric/generate_support');  %移除代码

%% brain script  手动修改 num 
for s = 1:6
%      eval(['s_',idc{s},'_p']) = zeros(1029,12);
%         eval(['s_',idc{s},'_p =  squeeze(brain_list(s,:,1:8));']);    % case 1
%      eval(['s_',idc{s},'_p =  squeeze(brain_list(s,:,1:4));']);    % case 1
%     eval(['s_',idc{s},'_p =  squeeze(brain_list(s,:,5:8));']);    % case 2
%     eval(['s_',idc{s},'_p =  squeeze(brain_list(s,:,9:12));']);  % case 3
    eval(['s_',idc{s},'_p =  squeeze(brain_list(s,:,1:12));']);  % all
end
%% from bst import cortex as cs
%  单独执行这部分
num = 2;%手动改， 1—6个算法：{'ori','re','mne','lor','sbl','foc'};
% eval(['cs.ImageGridAmp=s_',idc{num},'_p;']); 
% eval(['cs.ImageGridAmp(:,1:8)=s_',idc{num},'_p(:,1:8);']); 
% eval(['cs.ImageGridAmp(:,1:4)=s_',idc{num},'_p(:,1:4);']);
% eval(['cs.ImageGridAmp(:,1:12)=s_',idc{num},'_p(:,1:12);']); %all
eval(['cs.ImageGridAmp(:,9:12)=s_',idc{num},'_p(:,9:12);']); %case3
% eval(['cs.ImageGridAmp(:,9:12)=s_',idc{num},'_p(:,1:4)*1.5;']); %case3
% cs.ImageGridAmp(:,11)= cs.ImageGridAmp(:,11)* 1.5; %case3
% b_list=zeros(1029,12);
% b_list(:,1:8)=cs.ImageGridAmp;
% cs.ImageGridAmp=b_list;

%% 
% % cs.Time=1:12;
% % cs.Time=1:4;
% % es.Time = 1:4;
% es.F = es.F(:,1:4);
% 
% es.Time = 1:12;
% es.F = es.F(:,1:12);