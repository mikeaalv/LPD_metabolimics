% formulate the input data for LPD models
close all;
clear all;
workdir='/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/lpd/data/';
cd(workdir);
load('/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/ridge.net/result_reprod/result/quality_check/network_data.mat');
nfeatures=size(mat_reshape_all,2);
ntime=52;
% scaling to [0 1] and then discrete into [0 1 2] low middle high
mat_all_discr=[];
for isample=1:6
  rowind=((isample-1)*ntime+1):(isample*ntime);
  mat_samp=[];
  for ifeature=1:nfeatures
    int_temp_vec=mat_reshape_all(rowind,ifeature);
    int_temp_vec=(int_temp_vec-min(int_temp_vec))/(max(int_temp_vec)-min(int_temp_vec));
    disc_temp_vec=discretize(int_temp_vec,[0 1/3,2/3 1]);
    mat_samp=[mat_samp disc_temp_vec];
  end
  mat_all_discr=[mat_all_discr; mat_samp];
end
dlmwrite('input.csv',mat_all_discr);
