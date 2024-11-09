# GPU-MOT
Implementation of a high-performance Multi-Object Tracking (MOT) algorithm for NVIDIA embebbed boards.

## Usage
```bash
make MAX_KF=num_objects
./gpu_mot input_file show_screen
```

The code must be built by defining the variable ```MAX_KF```, which sets globally the maximum number of objects managed by the system (```num_objects```). Its value must be at most 1000.
When executing the program, the parameter ```input_file``` represents the scenario file to be used as input for the system.
A single line of the ```input_file``` must contain the following data: ```time, num_det, {dets}```.
The ```{dets}``` part of each line in turn contains sequential information about all the objects detected at that time instant, organized as follows: ```label, id, px, py, vx, vy, width, height```
The boolean parameter ```show_screen``` regulates if users will see the scenario evolution on-screen (1) or not (0).

## Example
```bash
make MAX_KF=100
./gpu_mot samples/2objs.csv 1
```

Where ```2objs.csv``` is a file about a synthetic scenario with 2 objects containing the following information:
```bash
0,2,3,0,180,66,5,5,10,12,3,1,558,60,-5,5,10,12
1,2,3,0,181,67,5,5,10,12,3,1,557,61,-5,5,10,12
2,2,3,0,181,67,5,5,10,12,3,1,557,61,-5,5,10,12
3,2,3,0,182,68,5,5,10,12,3,1,556,62,-5,5,10,12
...
```
The ```samples``` folder contains other sample files ready to use.

## Citation

If you use this repository in your work, please cite:
```BibTeX
@article{Medaglini2024,
  title={{High-Performance Multi-Object Tracking for Autonomous Driving in Urban Scenarios with Heterogeneous Embedded Boards}},
  journal={IEEE Access},
  author={Medaglini, A. and Peccerillo, B. and Bartolini, S.},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2024.3484129}
}
```
