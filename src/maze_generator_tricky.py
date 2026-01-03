#!/usr/bin/env python3
"""
Maze Generator using Recursive Backtracking (Depth-First Search)

This script generates random, solvable mazes and can visualize them
as ASCII art or save them as PNG images.
"""

import random
from typing import Optional
from pathlib import Path


class MazeGenerator:
    """Generate random solvable mazes using recursive backtracking."""
    
    def __init__(self, width: int = 21, height: int = 21, seed: Optional[int] = None, 
                 deceptive: bool = False, challenging: bool = False):
        """
        Initialize the maze generator.
        
        Args:
            width: Width of the maze (should be odd for proper walls)
            height: Height of the maze (should be odd for proper walls)
            seed: Random seed for reproducibility (None for random)
            deceptive: If True, create longer false paths (dead ends)
            challenging: If True, create a more confusing maze with balanced branches
        """
        # Ensure odd dimensions for proper wall structure
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.deceptive = deceptive
        self.challenging = challenging
        
        if seed is not None:
            random.seed(seed)
        
        # Initialize maze grid (1 = wall, 0 = path)
        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Start and end positions
        self.start = (1, 1)
        self.end = (self.height - 2, self.width - 2)
        
        # Solution path (filled after solving)
        self.solution_path: list[tuple[int, int]] = []
        
        # Track dead ends for deceptive mode
        self.dead_ends: list[tuple[int, int]] = []
    
    def generate(self) -> list[list[int]]:
        """Generate the maze using recursive backtracking."""
        if self.challenging:
            self._carve_passages_challenging()
        else:
            self._carve_passages(1, 1)
        
        # Ensure start and end are open
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.end[0]][self.end[1]] = 0
        
        # Find and store the solution
        self._solve()
        
        # If deceptive mode, extend the dead ends
        if self.deceptive:
            self._extend_dead_ends()
        
        return self.maze
    
    def _carve_passages_challenging(self) -> None:
        """
        Create a more challenging maze using modified Prim's algorithm.
        This creates more balanced branches that all look equally promising.
        """
        # Start from a random cell
        start_row, start_col = 1, 1
        self.maze[start_row][start_col] = 0
        
        # Frontier: walls that could be carved (with the cell they'd connect to)
        frontier = []
        
        def add_frontier(row, col):
            """Add neighboring walls to frontier."""
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                new_row, new_col = row + dr, col + dc
                wall_row, wall_col = row + dr // 2, col + dc // 2
                if (1 <= new_row < self.height - 1 and 
                    1 <= new_col < self.width - 1 and
                    self.maze[new_row][new_col] == 1):
                    frontier.append((wall_row, wall_col, new_row, new_col))
        
        add_frontier(start_row, start_col)
        
        while frontier:
            # Pick a random wall from frontier (this is key for balanced branches)
            idx = random.randint(0, len(frontier) - 1)
            wall_row, wall_col, new_row, new_col = frontier.pop(idx)
            
            # Check if the cell is still unvisited
            if self.maze[new_row][new_col] == 1:
                # Carve the wall and the cell
                self.maze[wall_row][wall_col] = 0
                self.maze[new_row][new_col] = 0
                
                # Add new frontier walls
                add_frontier(new_row, new_col)
    
    def _carve_passages(self, start_row: int, start_col: int) -> None:
        """Iteratively carve passages through the maze using a stack (no recursion limit)."""
        stack = [(start_row, start_col)]
        self.maze[start_row][start_col] = 0
        
        while stack:
            row, col = stack[-1]
            
            # Directions: up, right, down, left (moving by 2 to skip walls)
            directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
            random.shuffle(directions)
            
            carved = False
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check if the new cell is within bounds and unvisited
                if (1 <= new_row < self.height - 1 and 
                    1 <= new_col < self.width - 1 and
                    self.maze[new_row][new_col] == 1):
                    
                    # Carve through the wall between current and new cell
                    self.maze[row + dr // 2][col + dc // 2] = 0
                    self.maze[new_row][new_col] = 0
                    stack.append((new_row, new_col))
                    carved = True
                    break
            
            if not carved:
                # Dead end - backtrack
                self.dead_ends.append((row, col))
                stack.pop()
    
    def _would_create_open_area(self, row: int, col: int) -> bool:
        """Check if making this cell a path would create a 2x2 open area."""
        # Check all four 2x2 squares that include this cell
        for dr, dc in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            # Top-left corner of the 2x2 square to check
            r, c = row + dr, col + dc
            
            # Count open cells in this 2x2 area (treating the target cell as open)
            open_count = 0
            for check_r in range(r, r + 2):
                for check_c in range(c, c + 2):
                    if check_r < 0 or check_r >= self.height or check_c < 0 or check_c >= self.width:
                        continue
                    if (check_r, check_c) == (row, col):
                        open_count += 1  # This cell would become open
                    elif self.maze[check_r][check_c] == 0:
                        open_count += 1
            
            if open_count >= 4:
                return True
        
        return False

    def _extend_dead_ends(self) -> None:
        """Extend dead ends to create longer false paths."""
        solution_set = set(self.solution_path)
        
        # Find cells adjacent to the solution that could spawn false paths
        candidates = []
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if self.maze[row][col] == 0 and (row, col) in solution_set:
                    # Check for walls we could break through
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        wall_r, wall_c = row + dr, col + dc
                        beyond_r, beyond_c = row + 2*dr, col + 2*dc
                        
                        if (1 <= beyond_r < self.height - 1 and
                            1 <= beyond_c < self.width - 1 and
                            self.maze[wall_r][wall_c] == 1 and
                            self.maze[beyond_r][beyond_c] == 1):
                            candidates.append((row, col, dr, dc))
        
        # Create several false paths branching from solution
        random.shuffle(candidates)
        num_false_paths = min(len(candidates), max(3, (self.width * self.height) // 50))
        
        for i in range(num_false_paths):
            if i >= len(candidates):
                break
            row, col, dr, dc = candidates[i]
            self._carve_false_path(row, col, dr, dc, solution_set)
    
    def _carve_false_path(self, start_row: int, start_col: int, dr: int, dc: int, solution_set: set) -> None:
        """Carve a long winding false path from a point on the solution."""
        # Break through the initial wall
        wall_r, wall_c = start_row + dr, start_col + dc
        row, col = start_row + 2*dr, start_col + 2*dc
        
        if not (1 <= row < self.height - 1 and 1 <= col < self.width - 1):
            return
        if self.maze[row][col] == 0:  # Already a path
            return
        
        # Check if this would create a 2x2 open area
        if self._would_create_open_area(wall_r, wall_c) or self._would_create_open_area(row, col):
            return
        
        # Check that the new cell only connects back through our wall (not to other paths)
        open_neighbors = 0
        for check_dr, check_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            check_r, check_c = row + check_dr, col + check_dc
            if (0 <= check_r < self.height and 
                0 <= check_c < self.width and
                self.maze[check_r][check_c] == 0):
                open_neighbors += 1
        
        # Should only have 1 open neighbor (the wall we're about to carve connects to start)
        # But wall isn't carved yet, so should be 0
        if open_neighbors > 0:
            return
            
        self.maze[wall_r][wall_c] = 0
        self.maze[row][col] = 0
        
        # Now carve a winding path
        path_length = random.randint(5, 15)
        
        for _ in range(path_length):
            # Prefer to continue in same direction, but sometimes turn
            if random.random() < 0.6:
                directions = [(dr, dc), (-dc, dr), (dc, -dr)]  # forward, left, right
            else:
                directions = [(-dc, dr), (dc, -dr), (dr, dc)]  # left, right, forward
            
            carved = False
            for d_row, d_col in directions:
                new_row, new_col = row + 2*d_row, col + 2*d_col
                wall_row, wall_col = row + d_row, col + d_col
                
                if (1 <= new_row < self.height - 1 and
                    1 <= new_col < self.width - 1 and
                    self.maze[new_row][new_col] == 1 and
                    self.maze[wall_row][wall_col] == 1):
                    
                    # Check if this would create a 2x2 open area
                    if self._would_create_open_area(wall_row, wall_col) or self._would_create_open_area(new_row, new_col):
                        continue
                    
                    # Count how many open neighbors the new cell would have
                    # (excluding the wall we're about to carve)
                    open_neighbor_count = 0
                    for check_dr, check_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        check_r, check_c = new_row + check_dr, new_col + check_dc
                        if (check_r, check_c) == (wall_row, wall_col):
                            continue  # This is where we're coming from
                        if (0 <= check_r < self.height and 
                            0 <= check_c < self.width and
                            self.maze[check_r][check_c] == 0):
                            open_neighbor_count += 1
                    
                    # Only carve if this won't connect to any other paths
                    if open_neighbor_count == 0:
                        self.maze[wall_row][wall_col] = 0
                        self.maze[new_row][new_col] = 0
                        row, col = new_row, new_col
                        dr, dc = d_row, d_col
                        carved = True
                        break
            
            if not carved:
                break
    
    def _solve(self) -> bool:
        """Find the solution path using BFS."""
        from collections import deque
        
        start = self.start
        end = self.end
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (row, col), path = queue.popleft()
            
            if (row, col) == end:
                self.solution_path = path
                return True
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < self.height and 
                    0 <= new_col < self.width and
                    self.maze[new_row][new_col] == 0 and
                    (new_row, new_col) not in visited):
                    
                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), path + [(new_row, new_col)]))
        
        return False
    
    def to_ascii(self, show_solution: bool = False) -> str:
        """
        Convert maze to ASCII art.
        
        Args:
            show_solution: If True, mark the solution path with dots
        
        Returns:
            String representation of the maze
        """
        solution_set = set(self.solution_path) if show_solution else set()
        
        chars = {
            'wall': '██',
            'path': '  ',
            'solution': '··',
            'start': 'S ',
            'end': 'E ',
        }
        
        lines = []
        for row in range(self.height):
            line = ''
            for col in range(self.width):
                if (row, col) == self.start:
                    line += chars['start']
                elif (row, col) == self.end:
                    line += chars['end']
                elif self.maze[row][col] == 1:
                    line += chars['wall']
                elif (row, col) in solution_set:
                    line += chars['solution']
                else:
                    line += chars['path']
            lines.append(line)
        
        return '\n'.join(lines)
    
    def to_simple_ascii(self, show_solution: bool = False) -> str:
        """
        Convert maze to simple ASCII (works in all terminals).
        
        Args:
            show_solution: If True, mark the solution path with dots
        
        Returns:
            String representation of the maze
        """
        solution_set = set(self.solution_path) if show_solution else set()
        
        lines = []
        for row in range(self.height):
            line = ''
            for col in range(self.width):
                if (row, col) == self.start:
                    line += 'S'
                elif (row, col) == self.end:
                    line += 'E'
                elif self.maze[row][col] == 1:
                    line += '#'
                elif (row, col) in solution_set:
                    line += '.'
                else:
                    line += ' '
            lines.append(line)
        
        return '\n'.join(lines)
    
    def save_as_image(
        self, 
        filename: str = "maze.png",
        cell_size: int = 20,
        wall_color: tuple = (40, 40, 40),
        path_color: tuple = (255, 255, 255),
        solution_color: tuple = (100, 200, 100),
        start_color: tuple = (100, 100, 255),
        end_color: tuple = (255, 100, 100),
        show_solution: bool = False
    ) -> str:
        """
        Save the maze as a PNG image.
        
        Args:
            filename: Output filename
            cell_size: Size of each cell in pixels
            wall_color: RGB color for walls
            path_color: RGB color for paths
            solution_color: RGB color for solution path
            start_color: RGB color for start position
            end_color: RGB color for end position
            show_solution: If True, highlight the solution path
        
        Returns:
            Path to the saved file
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError("PIL/Pillow is required for image export. Install with: pip install Pillow")
        
        # Create image
        img_width = self.width * cell_size
        img_height = self.height * cell_size
        image = Image.new('RGB', (img_width, img_height), wall_color)
        draw = ImageDraw.Draw(image)
        
        solution_set = set(self.solution_path) if show_solution else set()
        
        # Draw cells
        for row in range(self.height):
            for col in range(self.width):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                if (row, col) == self.start:
                    color = start_color
                elif (row, col) == self.end:
                    color = end_color
                elif self.maze[row][col] == 0:
                    if (row, col) in solution_set:
                        color = solution_color
                    else:
                        color = path_color
                else:
                    color = wall_color
                
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Save image
        image.save(filename)
        return filename


def main():
    """Main function demonstrating maze generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random solvable mazes')
    parser.add_argument('-W', '--width', type=int, default=21, help='Maze width (default: 21)')
    parser.add_argument('-H', '--height', type=int, default=21, help='Maze height (default: 21)')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--deceptive', action='store_true', 
                        help='Create longer false paths (dead ends) to make the maze harder')
    parser.add_argument('--challenging', action='store_true',
                        help='Create a more confusing maze where all paths look equally promising')
    parser.add_argument('--solution', action='store_true', help='Show solution path')
    parser.add_argument('-o', '--output', type=str, default=None, help='Save as PNG image')
    parser.add_argument('--both', type=str, default=None, 
                        help='Save both puzzle and solution as PNG (provide base name, e.g., "maze" creates maze_puzzle.png and maze_solution.png)')
    parser.add_argument('--cell-size', type=int, default=20, help='Cell size in pixels for image output')
    parser.add_argument('--simple', action='store_true', help='Use simple ASCII characters')
    
    args = parser.parse_args()
    
    # Generate maze
    generator = MazeGenerator(width=args.width, height=args.height, seed=args.seed, 
                              deceptive=args.deceptive, challenging=args.challenging)
    generator.generate()
    
    # Output both puzzle and solution
    if args.both:
        try:
            base_name = args.both.replace('.png', '')  # Remove .png if provided
            
            puzzle_path = generator.save_as_image(
                filename=f"{base_name}_puzzle.png",
                cell_size=args.cell_size,
                show_solution=False
            )
            print(f"Puzzle saved to: {puzzle_path}")
            
            solution_path = generator.save_as_image(
                filename=f"{base_name}_solution.png",
                cell_size=args.cell_size,
                show_solution=True
            )
            print(f"Solution saved to: {solution_path}")
            
        except ImportError as e:
            print(f"Error: {e}")
            print("\nFalling back to ASCII output:\n")
            print("=== PUZZLE ===")
            if args.simple:
                print(generator.to_simple_ascii(show_solution=False))
            else:
                print(generator.to_ascii(show_solution=False))
            print("\n=== SOLUTION ===")
            if args.simple:
                print(generator.to_simple_ascii(show_solution=True))
            else:
                print(generator.to_ascii(show_solution=True))
    
    # Output single file
    elif args.output:
        try:
            filepath = generator.save_as_image(
                filename=args.output,
                cell_size=args.cell_size,
                show_solution=args.solution
            )
            print(f"Maze saved to: {filepath}")
        except ImportError as e:
            print(f"Error: {e}")
            print("\nFalling back to ASCII output:\n")
            if args.simple:
                print(generator.to_simple_ascii(show_solution=args.solution))
            else:
                print(generator.to_ascii(show_solution=args.solution))
    else:
        if args.simple:
            print(generator.to_simple_ascii(show_solution=args.solution))
        else:
            print(generator.to_ascii(show_solution=args.solution))
    
    print(f"\nMaze size: {generator.width}x{generator.height}")
    print(f"Start: S at {generator.start}")
    print(f"End: E at {generator.end}")
    print(f"Solution length: {len(generator.solution_path)} steps")


if __name__ == "__main__":
    main()
