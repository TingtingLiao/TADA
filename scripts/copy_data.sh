#!/usr/bin/env bash
sub_dir=$1

code_dir=/mnt/mmtech01/usr/liaotingting/projects/TADA
res_di=/mnt/mmtech01/usr/liaotingting/projects/data/tada/$sub_dir
target_dir=/share/liaotingting/projects/data/tada/$sub_dir


#ls $res_di | xargs -i{} mkdir -p $target_dir/{}/checkpoints
#ls $res_di | xargs -i{} cp -r $res_di/{}/checkpoints/geo_tex_ep0150.pth $target_dir/{}/checkpoints/

#ls $res_di | xargs -i{} mkdir -p $target_dir/{}/results
#ls $res_di | xargs -i{} cp -r $res_di/{}/results/. $target_dir/{}/results/

#ls $res_di | xargs -i{} mkdir -p $target_dir/{}/mesh
#ls $res_di | xargs -i{} cp -r $res_di/{}/mesh/. $target_dir/{}/mesh/

ls $res_di | xargs -i{} mkdir -p $target_dir/{}/validation
ls $res_di | xargs -i{} cp -r $res_di/{}/validation/geo_tex_ep0100.png $target_dir/{}/validation/
#ls $res_di | xargs -i{} cp -r $res_di/{}/validation/geo_ep0080.png $target_dir/{}/validation/

