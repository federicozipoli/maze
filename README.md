# maze
Generate mazes

## Basic maze (ASCII output)
`python maze_generator.py`

## Generate puzzle + solution pair in one command
`python maze_generator.py -W 31 -H 31 --both my_maze`

## Larger maze with solution shown
`python maze_generator.py -W 41 -H 41 --solution`

## Save as PNG image
`python maze_generator.py -W 31 -H 31 -o my_maze.png`

## Reproducible maze with a specific seed
`python maze_generator.py --seed 42`

## Use simple ASCII (# for walls) instead of Unicode blocks
`python maze_generator.py --simple`

## Generate a tricky maze with longer false paths
python maze_generator.py -W 31 -H 31 --deceptive --both my_hard_maze

