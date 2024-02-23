# GPU-MOT
Implementation of a high-performance Multi-Object Tracking (MOT) algorithm for NVIDIA embebbed boards.

## Usage
```bash
make MAX_KF=num_objects
./gpu_mot scenario_name
```

The code must be built by defining the variable ```MAX_KF```, which set globally the maximum number of objects managed by the system (```num_objects```).
When executing the program the parameter ```scenario_name``` represents the scenario file to be used as input for the system.
A single line of the ```scenario_name``` must contain the following data: ```time, num_det, {dets}```.
The ```{dets}``` part of each line in turn contains sequential information about all the objects detected at that time instant, organized as follows: ```label, id, px, py, vx, vy, width, height```

## Example
```bash
make MAX_KF=100
./gpu_mot 2.csv
```

Where ```2.csv``` is a file about a synthetic scenario with 2 objects and it contains the following kind of information:
```bash
0,2,1,0,180,66,5,5,10,12,3,1,558,60,-5,5,8,4
1,2,1,0,183,69,6,6,10,12,3,1,555,63,-6,6,8,4
2,2,1,0,206,92,11,11,10,12,3,1,532,86,-11,11,8,4
3,2,1,0,279,165,20,20,10,12,3,1,459,159,-20,20,8,4
...
```

## Citation

If you use this repository in your work, please cite:
```BibTeX
@inproceedings{XXXX,
  title={{}},
  author={},
  booktitle={},
  pages={},
  year={},
  doi={},
}
```
