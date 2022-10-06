tsv_dir=./datasets/genshin-20220915
feat_dir=./feat_dir
lab_dir=./lab_dir

nshard=2
n_cluster=128

# First, use simple_kmeans to generate labels

# feature extraction (MFCC feature)
for rank in $(seq 0 $((nshard - 1)))
do
python hubert/simple_kmeans/dump_mfcc_feature.py ${tsv_dir} train ${nshard} ${rank} ${feat_dir} --sample_rate 48000
python hubert/simple_kmeans/dump_mfcc_feature.py ${tsv_dir} valid ${nshard} ${rank} ${feat_dir} --sample_rate 48000
python hubert/simple_kmeans/dump_mfcc_feature.py ${tsv_dir} test ${nshard} ${rank} ${feat_dir} --sample_rate 48000
done

# k-means clustering
python hubert/simple_kmeans/learn_kmeans.py ${feat_dir} train ${nshard} km_train.joblib.pkl ${n_cluster} --percent -1 --max_iter 10000
python hubert/simple_kmeans/learn_kmeans.py ${feat_dir} valid ${nshard} km_valid.joblib.pkl ${n_cluster} --percent -1 --max_iter 10000
python hubert/simple_kmeans/learn_kmeans.py ${feat_dir} test ${nshard} km_test.joblib.pkl ${n_cluster} --percent -1 --max_iter 10000
# k-means application
for rank in $(seq 0 $((nshard - 1)))
do
python hubert/simple_kmeans/dump_km_label.py ${feat_dir} train km_train.joblib.pkl ${nshard} ${rank} ${lab_dir}
python hubert/simple_kmeans/dump_km_label.py ${feat_dir} valid km_valid.joblib.pkl ${nshard} ${rank} ${lab_dir}
python hubert/simple_kmeans/dump_km_label.py ${feat_dir} test km_test.joblib.pkl ${nshard} ${rank} ${lab_dir}
done

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/valid_${rank}_${nshard}.km
done > $lab_dir/valid.km

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/test_${rank}_${nshard}.km
done > $lab_dir/test.km


for x in $(seq 0 $((n_cluster - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt