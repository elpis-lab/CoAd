# COAD: Constant-Time Planning for Continuous Goal Manipulation with Compressed Library and Online Adaptation

Implementation of paper "COAD: Constant-Time Planning for Continuous Goal Manipulation with Compressed Library and Online Adaptation". This is a framework to provide constant-time solutions to goal-varying motion planning problems through a compressed library and fast online adaptation.

[Paper TBA] [Pre-print TBA] [Presentation Video TBA]

<p align="center">
    <img src="doc/intro.gif" width="600"/>
</p>

<p align="center">
    <!-- <a href="https://youtu.be/">
        <img src="doc/intro.gif" width="600"/>
    </a> -->
    <img src="doc/intro.jpg" width="600"/>
</p>


## Dependency
This repository is developed with python 3.10.12 in Ubuntu 22.

#### Major python dependencies
```
pip install -r requirements.txt
```

#### OMPL
Install OMPL python bindings with provided pre-built wheels in [OMPL Github Releases](https://github.com/ompl/ompl/releases). This project uses OMPL 1.7.0.

#### Real-robot Deployment
If you want to run this with a real UR robot

```bash
pip install -r requirements_robot.txt
```


## Run this project

<p align="center">
    <img src="doc/sim.jpg" width="600"/>
</p>

### Build library

### Comparison of Adaptation Methods

<p align="center">
    <img src="doc/grr.gif" width="600"/>
</p>

<p align="center">
    <img src="doc/dmp.gif" width="600"/>
</p>

<p align="center">
    <img src="doc/opt.gif" width="600"/>
</p>

<p align="center">
    <img src="doc/grr_sim.gif" width="600"/>
</p>

<p align="center">
    <img src="doc/dmp_sim.gif" width="600"/>
</p>

<p align="center">
    <img src="doc/opt_sim.gif" width="600"/>
</p>


### Benchmarking

#### Generate task set
```
python plan_load/generate_task_set.py --env table --robot panda
```

#### Generate joint goal set from task set
```
python plan_load/generate_joint_goal_set.py --env table --robot panda --ik neighbor
```

#### Generate solution set from joint goal set
```
python plan_load/generate_task_paths.py --env table --robot panda --ik neighbor --planner RRTConnect
```

#### Run condensation to compress solution set
```
python plan_load/condense_task_paths.py --env table --robot panda --ik neighbor --planner RRTConnect --adaptation linear
```

#### Benchmark results
```
python plan_load/
```

For detailed analysis and comparison, please refer to our paper.
