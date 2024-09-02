video1_path=data1.mp4
output_dir1=./output_data1
video2_path=data2.mp4
output_dir2=./output_data2

mkdir -p $output_dir1
mkdir -p $output_dir2

python splitvideo.py --video_path $video1_path --output_dir $output_dir1/frames
cd AlphaPose
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir ../$output_dir1/frames/ --outdir ../$output_dir1
cd ..

python infer_wild.py \
--vid_path $video1_path \
--json_path $output_dir1/alphapose-results.json \
--out_path $output_dir1/

python splitvideo.py --video_path $video2_path --output_dir $output_dir2/frames
cd AlphaPose
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir ../$output_dir2/frames/ --outdir ../$output_dir2
cd ..

python infer_wild.py \
--vid_path $video2_path \
--json_path $output_dir2/alphapose-results.json \
--out_path $output_dir2/
