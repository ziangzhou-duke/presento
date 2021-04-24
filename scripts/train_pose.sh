python pose/main.py --epochs 200 --step_size 150 --add_body_dnn --use_cnn_features --num_classes 7 \
  --num_total_iterations=1 --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12 --split_branches --do_fusion \
  --add_whole_body_branch --exp_name "HMT-4"