python src/openpose_3dpose_sandbox.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1 --load 24371 --pose_estimation_json ./test/t-sep --gif_fps 24


python src/openpose_3dpose_sandbox.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1 --load 24371 --pose_estimation_json ./test/t-sep/ --write_gif --gif_fps 24 --interpolation --multiplier 0.5