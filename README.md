# Planning with Goal Regions
Experience-based planner with goal regions.  

* generate_DS_top1_mj.py/generate_DS_box1_mj.py generates the entire data structure over a specified object distribution for the free and box case respectively (can also condense)
* condense_DS_top1_mjc.py/condense_DS_box1_mj.py condenses an existing data structure to a smaller collection of root paths
* load_DS_top1_mjc.py/load_DS_box1_mj.py loads a condensed data structure and samples problems, mostly for visualization
* benchmarking scripts sample a given number of problems and track success rates and solution times.

* Condensation currently has an early exit at 50% (to avoid running for too long)
* All helper functions and classes are in utils/ src/ and planning_mj.py
