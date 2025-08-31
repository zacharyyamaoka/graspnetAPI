
## Grasp Net

Well it hasn't been as easy as I thought!

- The naming conventinons they use are different and have a strange grasp frame (z going out sideways, palm offset by 2 cm, etc.)
- The collisions masks on the labels don't seem to work. Grasps in collision are returned from getLabels, when I would just except positive examples


Fixing Collision Label Issue:

1. Redownload data. 

    - There is mabye hints the data has been corrupted. Error msg: BadZipFile: Bad CRC-32 for file 'offsets.npy'
    - This could be caused by juypter.
    - It may be smart to keep another copy of the labels on hand in case the orginals get corrupted
    - The labels are fairly small compared to the images, which is great
    - I can try to download from Baidu, if perhaps the google ones are outdated (Update: tried this but couldn't download from Baidu)
    - Only 88 Grasp labels but 190 Collision labels, as grasp labels are associate with 88 models, and are then masked by the scene

2. Generate it agian
    - This is probably the better long term solution. Its a bad idea to have this offline label dataset
    that you do not know how it was created. 



### Learnings

I have learned alot by being able to just go through their code. I get confidence seeing where we have done similar things, and curosity were we differ.

- Ideas on visualization
    - Using thin grippers in o3d
- Methods of evaluation
    - Top K
    - Generating offline labels

### Generating Labels

https://github.com/GouMinghao/rgb_matters/blob/main/rgbd_graspnet/data/utils/gen_label.py
In RGB Matters, then generate the labels by iterating through all the ground truth, 

- They first align the rotation matrix to the towards vector in plane rotation
- Then they bin the position along the grid of the depth camera
- In this way, for each point and angle then can tell you if there is a grasp there...

- Thats very interesting! If I do fast anayltical searching and push forward in Z, it may end up going into a different grid cell
- That is actually ok... if you are consitent, then it will learn to rate grasps high, from where if you push forward it will suceed


---

"Avoid academic discussions, what is the simplest, fastest way to achieve this?" - Keenan

### Thoughts

This is a very cool dataset. A big problem with sim, is that the synthetic depth and color images are not realistic! this dataset
combines the benefits of sim ground truth with real data.

The key contribution is the hand labeling done to associate virtual 3D models with the real depth clouds. That allows you to analytically 
calculate if a grasp will succeed.

See MIT for step by step notes on calculating antipodal grasps: https://manipulation.csail.mit.edu/index.html

Dexnet is what is implemented and does something in a similar spirt. Contact model is described in dexnet 1: https://berkeleyautomation.github.io/dex-net/#dexnet_1

This of course is all just an approximation for the actual real dataset I want to use.

- The purpose of this dataset is not to really train a model. but to use as common sense check, and baseline.
- With fully dense annotations, you can rapidly evaluate loss (each images provides full supervision instead of just a few grasps)

Its also helpful to see how your models are performing on common bench marks used in industry

See 2025 paper on large-scale grasping datasets: [GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes](https://arxiv.org/abs/2504.06866)

https://sites.google.com/view/graspclutter6d




### Grasping Frame 
## Design Notes

What do I need to do in here?

- [ ] Make sure I have a clean way to access all the data in the scene
- [x] Color
- [x] Depth
- [x] Mask
- [x] Table
- [x] Object Pose
- [x] Object Mesh
- [ ] I want a way to evaluate if a grasp pose is good...

What can I do outside?

- [ ] Use open3D to create table
- [ ] Implement the Gym Env
- [ ] Tested out baseline algortihim
- [ ] Create height map function

How do I imagine using this?

Create Gym Env: the cool thing is I can use this gym for a bunch of different enviornments!

- On reset:
    - it returns a synethic depth image for the average + std of the table
    - it returns a pose for the table (I can use this to check vs the one that is calculated)
    - It returns the step() observation
- On Step:
    - given the last observation that was returned (scene, camera, annotation id) it checks if sucess
    - Sucess function should take a list of grasp poses, friction setting and then return true or false.
    - It goes to the next observation (random seeded or +1 index)
    - it returns the color, depth and mask image (just as would happen with the real robot/gazebo)
- On render:
    - it should show the poses that were sent, in an interactive open3d viewer? (you can manually set the sleep time)
    - Camera Pose should be set to be the actual camera pose.
    - Option to display the actual meshes
    - Option to display table
    - Option to show the nearest succesful grasp it was associate with?
    - I should see the finger width, etc. check for collisions, visually see if it succeeds or fails to make sure its working
    - https://www.open3d.org/docs/0.9.0/tutorial/Advanced/non_blocking_visualization.html

---

Another way is to rapidly learn, if you have a discrete action map, you can update all the values at the same time! instead of just 1-4

This will actually be great for testing! its a static baseline. easy to download, etc.

- I can also check that the algorithim is picking up the right label of object... and assert if it doesn't... then your selection code is wrong.

Testing....

Lets say it fails then what? How to prevent regression?

A function that outputs a heatmap... or evluates the goodness of a grasp (different approaches!)


HeatMap (Discretize action space)

Grasp Net 1B
    - Put all grasps into a list, loop through all the scenes (if input was stacked), and evaluate the grasps


Offline Bam Data
    - Find the action in the heatmap that is most similar to the action that was attempted (flag for exact/thresholds)
    - In case of failure, check that the value is low, in case of success, the value should be high


In Case of real Failure
    - Unit Test: Rerun on image, and make sure that the confidence for the grasp near the failure is low
    - How to learn success though? You cannot replay the exact scene...
    - Set up other similar real life scenarios (no guarentee you will find success, takes time, etc)
    - Create a synthetic scene manually using objects (perhaps a network can do this and then be fined tuned by human) and use dexnet eval
        - Cool thing is you could use same point cloud...
        - How to deal with deformed objects? Its cool beacuse you are getting the benefits of sim (analytical eval) + the benefits of real (real noisy data)
    - Manual annotate by humans about what success/bad looks like (Human annotate may be wrong though! and is costly) (you can use same image though)

    - deformed object prediction: https://repositum.tuwien.at/bitstream/20.500.12708/195069/1/Eder%20Christian%20-%202024%20-%20Pose%20Estimation%20of%20Deformable%20Objects.pdf
    - Could mabye train it in unsupervised way with all the data
    - Even just approximating scene geometry with primiative shapes...? lets assume it all becomes one solid object What if you miss grasps?
    - Could I even just get some successes though? let say I put in a primiative shsape, a table, etc. that hsould provide at least some succesful grasps...
    - Then evalaute heatmap... if there is no object near by, it may mean that it wasn't label, so skip.. you cannot say anything about it.
    - Simplest is to just try agian. and keep this failure in the dataset, and remember that any points near this should be considered a failure!
    - Humans are expensive, data is cheap. Just continue sorting trash, you will continueing running into failures, successes, it all goes into a replay buffer
    - If you change your action space alot then yes you may need to change the replay buffer... 

    - ^ this is completely bad. If you had an algorithim that could accurately estimate 6D pose, then you wouldn't need to grasp!!!
    
Sample + Score

- Sample and score is cool, beacuse you never have an issue of the graspsing representation being different...
- Ultiatemly perhaps you can try both lol, if you need to compress knowledge from one dataset to another one.

Object Detection

In Case of real Failure:
    - Classical tesla dataflywheel
    - Correctly annotate the failed case, as use that as ground truth unit test to prevent regression
    - Collect and annotate more similar cases



## Thoughts
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

I am not expecting any upstream changes... no meaningful changes in last 4 years.

I think its nice that its kept seperate, but I don't think I should feel bad about importing some select functions

I would prefer If I don't need to important anything from BAM into this though. better to just import the other way...
I can copy paste functions if needed.

## Ok what do I want to achieve now?

I would prefer.

Let me quickly make the gym env...

### Eval

You should except a delay every time you load a new scene as it needs to read in models from file
and voxelize them, and they are very dense pointclouds!

### Tests

There is a function that checks all the dataset is loaded!


### Installation

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
python3 -m pip install -e /home/bam/graspnetAPI 
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