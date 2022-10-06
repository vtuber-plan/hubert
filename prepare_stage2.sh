tsv_dir=./datasets/genshin-20220915
feat_dir=./feat_dir
lab_dir=./lab_dir
ckpt_path=./None/checkpoints/checkpoint_best.pt

nshard=1
rank=0
n_cluster=100
layer=12

# feature extraction (HUBERT feature)
python hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} genshin_train ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
python hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} genshin_valid ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
python hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} genshin_test ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
