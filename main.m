close all; clear all; clc

%access all names: meta(:).filename  
%access one name: meta(5).filename
%access one self score: meta(5).skill_self
%access corresponding kinematics values: K(5)
%access transcriptions: T(5)

%add all subfolders to path
folder = fileparts(which(mfilename)); 
addpath(genpath(folder));

%% Import meta stuff as a structure array
meta = readtable('JIGSAWS/Suturing/meta_file_Suturing.txt');
meta = table2struct(meta);
meta = rmfield(meta, 'Var2');
meta = rmfield(meta, 'Var5');
meta = cell2struct( struct2cell(meta), {'filename', 'skill_self', 'skill_GRS', 'Score1','Score2','Score3','Score4','Score5','Score6'});
num_surgeons = size(meta,1);

%% Import kinematics data
K = containers.Map('KeyType','double','ValueType','any');
i = 1;
for name = {meta(:).filename}
   disp('Importing kinematics: ' + string(name));
   K(i) = importdata('JIGSAWS/Suturing/kinematics/AllGestures/'+string(name)+'.txt');
   i = i + 1;
end

%% Import transcriptions
T = containers.Map('KeyType','double','ValueType','any');
i = 1;
for name = {meta(:).filename}
   disp('Importing transcriptions: ' + string(name));
   T(i) = table2cell(readtable('JIGSAWS/Suturing/transcriptions/'+string(name)+'.txt'));
   i = i + 1;
end
%% Visualize something

%plot one kinematics of one gesture together

Gesture = 'G' + string(1);
xyz_l = [];
vel_l = [];

for i = 1:num_surgeons
   if(meta(i).skill_self == 'E')
       Kinematics_i = K(i);
       Transcription_i = T(i);
       Selection = cell2mat(Transcription_i(cell2mat(Transcription_i(:,3)) == Gesture,1:2));
       frame_start = Selection(:,1);
       frame_end = Selection(:,2);
       
       for j = 1:size(frame_start,1)
           xyz_l = [xyz_l;Kinematics_i(frame_start(j):frame_end(j),1:3)];
           vel_l = [vel_l;Kinematics_i(frame_start(j):frame_end(j),13:15)];
       end
   end
   
   
   
   
   
   
end




% N = [];
% I = [];
% E = [];
% for i = 1:num_surgeons
%     if meta(i).skill_self == 'N'
%         N = [N, meta(i).skill_GRS];
%     end
%     if meta(i).skill_self == 'I'
%         I = [I, meta(i).skill_GRS];
%     end
%     if meta(i).skill_self == 'E'
%         E = [E, meta(i).skill_GRS];
%     end
% end
% 
% hAxes = axes('NextPlot','add',...           %# Add subsequent plots to the axes,
%              'DataAspectRatio',[1 1 1],...  %#   match the scaling of each axis,
%              'XLim',[0 30],...               %#   set the x axis limit,
%              'YLim',[0 eps],...             %#   set the y axis limit (tiny!),
%              'Color','none');               %#   and don't use a background color
% plot(N,0,'r+',I,0,'gx',E,0,'bo','MarkerSize',10);  %# Plot data set 1
% hold on
% title("Self Proclaimed Skill vs GRS Skill");
% %legend('Novice','Intermediate','Expert');
