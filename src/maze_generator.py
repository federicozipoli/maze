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
    
    def __init__(self, width: int = 21, height: int = 21, seed: Optional[int] = None):
        """
        Initialize the maze generator.
        
        Args:
            width: Width of the maze (should be odd for proper walls)
            height: Height of the maze (should be odd for proper walls)
            seed: Random seed for reproducibility (None for random)
        """
        # Ensure odd dimensions for proper wall structure
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        
        if seed is not None:
            random.seed(seed)
        
        # Initialize maze grid (1 = wall, 0 = path)
        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Start and end positions
        self.start = (1, 1)
        self.end = (self.height - 2, self.width - 2)
        
        # Solution path (filled after solving)
        self.solution_path: list[tuple[int, int]] = []
    
    def generate(self) -> list[list[int]]:
        """Generate the maze using recursive backtracking."""
        self._carve_passages(1, 1)
        
        # Ensure start and end are open
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.end[0]][self.end[1]] = 0
        
        # Find and store the solution
        self._solve()
        
        return self.maze
    
    def _carve_passages(self, row: int, col: int) -> None:
        """Recursively carve passages through the maze."""
        self.maze[row][col] = 0
        
        # Directions: up, right, down, left (moving by 2 to skip walls)
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        random.shuffle(directions)
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if the new cell is within bounds and unvisited
            if (1 <= new_row < self.height - 1 and 
                1 <= new_col < self.width - 1 and
                self.maze[new_row][new_col] == 1):
                
                # Carve through the wall between current and new cell
                self.maze[row + dr // 2][col + dc // 2] = 0
                self._carve_passages(new_row, new_col)
    
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
    parser.add_argument('--solution', action='store_true', help='Show solution path')
    parser.add_argument('-o', '--output', type=str, default=None, help='Save as PNG image')
    parser.add_argument('--both', type=str, default=None, 
                        help='Save both puzzle and solution as PNG (provide base name, e.g., "maze" creates maze_puzzle.png and maze_solution.png)')
    parser.add_argument('--cell-size', type=int, default=20, help='Cell size in pixels for image output')
    parser.add_argument('--simple', action='store_true', help='Use simple ASCII characters')
    
    args = parser.parse_args()
    
    # Generate maze
    generator = MazeGenerator(width=args.width, height=args.height, seed=args.seed)
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
