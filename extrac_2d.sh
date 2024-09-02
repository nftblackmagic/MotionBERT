video_path=123.mp4

python splitvideo.py --video_path 1235.mp4 --output_dir frames
cd AlphaPose
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir ../frames/ --save_img --outdir ../2d/
cd ..

python infer_wild.py \
--vid_path 1235.mp4 \
--json_path ./2d/alphapose-results.json \
--out_path ./2d/