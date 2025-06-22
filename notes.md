

Docs: https://graspnetapi.readthedocs.io/en/latest/index.html

- 190 Scenes
- 100 for Training, 90 for Testing (30 seen, 30 similar, 30 novel)
- 88 Objects
- 256 Annotations per scene (per camera)
- about 97,280 RGBD images in total then
- 3M - 9M Grasp poses per scene... thats crazy!

- How to evaluate correctness?


How to interact with the api?

Its very useful to look at this function: loadScenePointCloud() and look at src code graspnetAPI/graspnet.py

It shows basically how to do all the things you would need to do...



### Tests

There is a function that checks all the dataset is loaded!


### Installation

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

How to clear jupyter notebook

In vscode select python interperter
run script to open terminal

```bash
python -m pip install nbstripout
nbstripout --install
```

## Change Log

#### 1.2.6

-

#### Pre June 22

- Added loadCameraK() to access intrinsics directly
- Change path of graspnet_root to local path