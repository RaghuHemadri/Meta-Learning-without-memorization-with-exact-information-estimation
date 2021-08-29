# Meta Learning without memorization with exact information estimation
*This repository is improvision over [Meta-Learning without Memorization](https://openreview.net/pdf?id=BklEFpEYwS) by Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, Chelsea Finn.*

## Project Overview
### Meta-Learning without Memorization Summary
This paper analyses a pitfall of current meta-learning algorithms, where the task can be inferred from the meta-training data alone, leaving the task-training data unused. Such a meta-learner would generalise well on the meta-training tasks, but will fail to generalise on new tasks at test time. This kind of overfitting is formalised as the memorization problem. This problem is implicitly resolved in current meta-learning algorithms by constructing mutually-exclusive meta-training tasks, which is not easy to construct in all scenarios. The paper introduces an information-theoretic meta-regularizer which forces information extraction from the task data (D) by restricting information flow from meta-parameters (&theta;) and input (x*). Experimental evaluation with one gradient based and one contextual meta-learning method, on non-mutually-exclusive tasks bring out the mettle of the proposed regulariser. 

### My contribution
The authors of [Meta-Learning without Memorization](https://openreview.net/pdf?id=BklEFpEYwS) encourage the information in data D to be applied in the prediction of yˆ*
 by restricting the information from input x* and meta-parameters &theta;. Alternatively, this repository directly maximizes the mutual information I(y*; D|x*, θ) in Definition 1, using recent MI estimators such as [MINE](https://arxiv.org/abs/1801.04062). MINE is a Mutual Information Neural Estimator (MINE) that is linearly scalable in dimensionality as well as in sample size, trainable through back-prop, and strongly consistent.

## Implementation
### Generating pose regression dataset
Requirements:
* TensorFlow (see tensorflow.org for how to install)
* numpy-stl
* gym
* mujoco-py

Step 1: Download CAD models from [Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild](http://cvgl.stanford.edu/projects/pascal3d.html) [ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip) and use the CAD folder.

We removed the two classes 'bottle' and 'train' because the objects are symmetric.

Step 2: Convert the CAD models from \*.off to \*.stl.

You can download a converter from [https://www.patrickmin.com/meshconv](https://www.patrickmin.com/meshconv). Then, run

```
chmod 755 meshconv
find ./CAD -maxdepth 2 -mindepth 2 -name "*.off" -exec meshconv -c stl {} \;
```

Step 3: Render the dataset. Using the utilities in pose_data
```
CAD_DIR=
DATA_DIR=
python mujoco_render.py --CAD_dir=${CAD_DIR} --data_dir=${DATA_DIR}
cp -r ${DATA_DIR}/rotate ${DATA_DIR}/rotate_resize
python resize_images.py --data_dir=${DATA_DIR}/rotate_resize
python data_gen.py --data_dir=${DATA_DIR}/rotate_resize
```
This generates two pickle files: train_data.pkl and val_data.pkl.

### Train models on pose regression dataset
See `pose_code/run.sh` for examples of training the various algorithms.
