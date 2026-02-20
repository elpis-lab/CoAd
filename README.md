# PlanLOAD 

## Dependency



## Run

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
