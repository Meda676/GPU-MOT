# GPU-MOT
Implementation of a high-performance Multi-Object Tracking (MOT) algorithm for NVIDIA embebbed boards.

## Usage
```bash
make MAX_KF=num_objects
./gpu_mot input_file show_screen
```

The code must be built by defining the variable ```MAX_KF```, which sets globally the maximum number of objects managed by the system (```num_objects```).
When executing the program, the parameter ```input_file``` represents the scenario file to be used as input for the system.
A single line of the ```input_file``` must contain the following data: ```time, num_det, {dets}```.
The ```{dets}``` part of each line in turn contains sequential information about all the objects detected at that time instant, organized as follows: ```label, id, px, py, vx, vy, width, height```
The boolean parameter ```show_screen``` allow users to see the scenario evolution (1) or not (0)

## Example
```bash
make MAX_KF=100
./gpu_mot data/2objs.csv 1
```

Where ```2objs.csv``` is a file about a synthetic scenario with 2 objects containing the following information:
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
