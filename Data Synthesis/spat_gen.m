% 	Date: 2020/3/3
%   Author: Gexin Huang
%   Function: generate spatial components of brain signals.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	-Input:
%   sample_num: the total number of generate samples
%   SourceNum: the number of sources
%   Vert: the vertex number of cortex space	
%	path_flag: 0 means the simulation data, 1 means the real data.
% 	mode: use to specify the conditions for simulation experiments. 
% 		mode 1  :different SNR of scalp
% 		mode 2  :different source area for single source patch
% 		mode 3  :different source area for two different source patch with different ERP
% 		mode 4  :different source area for three different source patch with different ERP
% 	varargin: 'count' use to generate batched samples.
%   -Output: cell[num,K], save the dictionary of activated source verteces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=spat_gen(sample_num,SourceNum,Vert,mode,path_flag,varargin)
     % variable argument output list 
     % variable argument input list 
% spat_gen(sample_num,,SourceNum,Vert,count)
% count: the parameter which determines how many batch the data set want to generate 
% varargin{1}=count
if isempty(varargin)
    count = 0;
else 
    count = varargin{1};
end
addpath('./support_func')
%% vertices 1024/3002/6004
if path_flag == 0
% simulation 
load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/Gain_l2_',int2str(Vert),'.mat']);
load(['./support_data/Gain_model_Colin_27_',int2str(Vert),'/Cortex_',int2str(Vert),'.mat']);
elseif path_flag == 1
% real
load(['./support_data/real_data_',int2str(Vert),'/Gain_n_',int2str(Vert),'.mat']);
load(['./support_data/real_data_',int2str(Vert),'/Cortex_',int2str(Vert),'.mat']);
end 

%% parameters
K = SourceNum; %source num
Cortex = Cortex;  % information matrix of brain anatomical structure.

%% mode setting for simulation experiments
%%%%%%% generate mode %%%%%%%%%%%%
if mode == 1 
    K = 1;  %sing
   % le source
    rand_area= [0,0]; %fix the source activity area 
elseif mode == 2
    K = 1;  %single source
elseif mode == 3
    K = 2;  %double source
elseif mode == 4
    K = 3;  %triple source
end   
[~, VertArea] = tess_area(Cortex);  %% generate area 
dic=cell(sample_num,K);
%% modify part : uniform lamb 
% way 1
% lamb = cell(sample_num,K);
% for j = 1:K
%     eval(['lamb',num2str(j),' = randfixedsum(4,double(sample_num),1,0,1)''',';'])
%     for i = 1:sample_num
%         eval(['lamb{i,',num2str(j),'}=lamb',num2str(j),'(i,:);'])
%     end
% end
% way 2: 再做一个正负抽样赋值
for j = 1:K
    eval(['lamb',num2str(j),' = randfixedsum(4,double(sample_num),1,0,1)''',';'])
    sig= 2*randi([0 1],[double(sample_num),4])-1;
    eval(['lamb',num2str(j),'=lamb',num2str(j),'.*sig;'])
    for i = 1:sample_num
        eval(['lamb{i,',num2str(j),'}=lamb',num2str(j),'(i,:);'])
    end
end


%% generate the activated source verteces
except_=0;  %count overlap
for i = 1:sample_num
  while true
%     seedvox = round(unifrnd (1,size(Gain_matrix,2),[1,K]));
   %% random area 
    if mode == 2 || mode == 1  % case of single source 
        seedvox = round(unifrnd (1,size(Gain_matrix,2),[1,K]));
        AreaDef = (1+0.2*randi(rand_area,1))*1e-3*ones(numel(seedvox),1); 
    elseif mode == 3 || mode == 4           % case of multiple sources 
        % check if the seedvox location is larger than interaction 
        while true
            seedvox = round(unifrnd (1,size(Gain_matrix,2),[1,K]));
            flag_1 = 0;
            for o = 1:K-1
                for j = o+1:K
                    if abs(seedvox(o)-seedvox(j))<100
                        flag_1 = 1;
                        break
                    end
                end
                if flag_1 == 1
                   break
                end
            end
            if flag_1 ==0
                break %都满足条件时，跳出循环
            end
        end
        % 
        AreaDef = ones(numel(seedvox),1); 
        for j = 1:K
            eval([' s',num2str(j),'= randi([0,15]); '])   % [0,20]
            eval(['AreaDef(',num2str(j),') = s',num2str(j),'* 1e-4;'])
        end
    end
   %% generate part 
    flag=0;   % the flag of intersection
    ActiveVoxSeed = num2cell(seedvox); %[k,1] vector
        for k = 1:numel(seedvox)
            ActiveVoxSeed{k} = PatchGenerate(seedvox(k),Cortex.VertConn,VertArea,AreaDef(k));
            dic{ (i-1)*simalar_sample+j , k }=ActiveVoxSeed{k};         
        end

    % check if any vertex in different sources intersected 
    %% Permutation and combination cycles, in which each source is compared to a different source   
        for k = 1:(K-1)
            temp_1=ActiveVoxSeed{k};
            t1num=numel(ActiveVoxSeed{k});
            for l = (k+1):K
                temp_2=ActiveVoxSeed{l};
                t2num=numel(ActiveVoxSeed{l});
                temp_in=intersect(temp_1,temp_2);   % find intersection sets
                intnum=numel(temp_in);
                inter_percent =intnum/(t1num+t2num-intnum)*100;
                if intnum ~= 0
                    flag=1;
                    except_=except_+1;
                    break               
                end
            end
            if flag==1  %jump out check sample
                break
            end
        end
        if flag == 1  %jump out simalar sample
            break
        end
   if flag==0
      break
   end
  end

end
except_   %output the count of intersection
%% Batch saving
if count == 0
    save('./syn_dara/area_dic.mat','dic');
else 
    eval(['dic_',int2str(count),'=dic']);
    save(['./syn_dara/area_dic_',int2str(count),'_.mat'],['dic_',int2str(count)]);
    eval(['lamb_',int2str(count),'=lamb;']);
    save(['./syn_dara/area_lamb_',int2str(count),'_.mat'],['lamb_',int2str(count)]);
    
%% set the Area of source as return
varargout{1}= AreaDef;
% plot test
% scatter3(Cort_8195.Vertices(:,1),Cort_8195.Vertices(:,2),Cort_8195.Vertices(:,3),'k')

end