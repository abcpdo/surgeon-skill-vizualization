close all; clear all

%access all names: meta(:).filename  
%access one name: meta(5).filename
%access one self score: meta(5).skill_self
%access corresponding kinematics values: kinematics(5)

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
kinematics = containers.Map('KeyType','double','ValueType','any');
i = 1;
for name = {meta(:).filename}
   disp('Importing: ' + string(name));
   kinematics(i) = importdata('JIGSAWS/Suturing/kinematics/AllGestures/'+string(name)+'.txt');
   i = i + 1;
end

%% Visualize something
N = [];
I = [];
E = [];
for i = 1:num_surgeons
    if meta(i).skill_self == 'N'
        N = [N, meta(i).skill_GRS];
    end
    if meta(i).skill_self == 'I'
        I = [I, meta(i).skill_GRS];
    end
    if meta(i).skill_self == 'E'
        E = [E, meta(i).skill_GRS];
    end
end

hAxes = axes('NextPlot','add',...           %# Add subsequent plots to the axes,
             'DataAspectRatio',[1 1 1],...  %#   match the scaling of each axis,
             'XLim',[0 30],...               %#   set the x axis limit,
             'YLim',[0 eps],...             %#   set the y axis limit (tiny!),
             'Color','none');               %#   and don't use a background color
plot(N,0,'r.',I,0,'g.',E,0,'b.','MarkerSize',10);  %# Plot data set 1
hold on
legend('Novice','Intermediate','Expert');
