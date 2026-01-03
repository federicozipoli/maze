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
`python maze_generator.py -W 31 -H 31 --deceptive --both my_hard_maze`


## Even more tricky
`python maze_generator_tricky.py -W 31 -H 31 --challenging --both my_hard_maze`
`python maze_generator_tricky.py -W 31 -H 31 --challenging --deceptive --both my_hard_maze`

The --challenging mode uses Prim's algorithm instead of recursive backtracking. The key difference is that it picks randomly from ALL frontier walls, not just the most recent one. This creates a more "bushy" maze where every branch looks equally promising.
You can also combine them: --challenging --deceptive for maximum confusion!

## Summary

### Basic maze
`python maze_generator.py -W 31 -H 31`

### Generate puzzle + solution pair
`python maze_generator.py -W 31 -H 31 --both my_maze`

### More confusing (all paths look equal)
`python maze_generator.py --challenging --both my_maze`

### Add longer dead ends
`python maze_generator.py --deceptive --both my_maze`

### Maximum difficulty
`python maze_generator.py --challenging --deceptive --both my_maze`

### Reproducible with seed
`python maze_generator.py --seed 42 --both my_maze`

