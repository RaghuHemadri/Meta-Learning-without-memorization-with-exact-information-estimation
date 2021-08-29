##!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r meta_learning_without_memorization/requirements.txt
pip install tensorflow-probability

cd meta_learning_with_memorization/pose_code
python create_fake_data.py
cd ../..

python -m meta_learning_without_memorization.pose_code.maml_bbb --metatrain_iterations=2 --data_dir=meta_learning_without_memorization/pose_data --data=train_data.pkl,val_data.pkl
#python -m meta_learning_without_memorization.pose_code.maml_vanilla --metatrain_iterations=2 --data_dir=meta_learning_without_memorization/pose_data --data=train_data.pkl,val_data.pkl
#python -m meta_learning_without_memorization.pose_code.np_all_bbb --num_updates=2 --data_dir=meta_learning_without_memorization/pose_data --data=train_data.pkl,val_data.pkl
#python -m meta_learning_without_memorization.pose_code.np_bbb --num_updates=2 --data_dir=meta_learning_without_memorization/pose_data --data=train_data.pkl,val_data.pkl
#python -m meta_learning_without_memorization.pose_code.np_ib --num_updates=2 --data_dir=meta_learning_without_memorization/pose_data --data=train_data.pkl,val_data.pkl
#python -m meta_learning_without_memorization.pose_code.np_vanilla --num_updates=2 --data_dir=meta_learning_without_memorization/pose_data --data=train_data.pkl,val_data.pkl
