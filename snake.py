import os
import sys
import random
import pygame
from collections import deque
from typing import Tuple, List
import numpy as np
import nn

Cell = Tuple[int, int]


class Game:
    def __init__(self, grid_size: int = 20, cell_size: int = 20, render: bool = True, fps: int = 10, seed: int = 0):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        self.render_enabled = render
        self.fps = fps

        # set seed for food spawning
        self._food_rng = random.Random(seed)

        self._clock = None
        self._screen = None
        if self.render_enabled:
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Darwin's Snake")
            self._clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.snake: deque[Cell] = deque()
        # start with length 3 going right
        self.snake.append((mid - 1, mid))
        self.snake.append((mid, mid))
        self.snake.append((mid + 1, mid))
        self.direction = (1, 0)
        self.grow = 0
        self.score = 0
        self.frame = 0
        self.done = False
        self.spawn_food()
        return self.get_state()

    def spawn_food(self):
        free = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.snake]
        if not free:
            self.food = None
            return
        
        if hasattr(self, "_food_rng"):
            self.food = self._food_rng.choice(free)
        else:
            self.food = random.choice(free)

    def get_state(self):
        return {
            "snake": list(self.snake),
            "food": self.food,
            "direction": self.direction,
            "score": self.score,
        }
    
    def get_input_vector(self) -> List[float]:
        # danger_straight = 0, 1
        # danger_left = 0, 1
        # danger_right = 0, 1
        # food_straight = 0->1
        # food_left = 0->1
        # food_right = 0->1
        # distance_straight_wall = 0->1
        # distance_left_wall = 0->1
        # distance_right_wall = 0->1

        head = self.snake[-1]
        dx, dy = self.direction
        
        # Get relative directions: straight, left, right
        straight = (dx, dy)
        left = (-dy, dx)
        right = (dy, -dx)
        
        directions = [straight, left, right]
        vector = []
        
        for direction in directions:
            # check danger (collision with wall or self)
            next_pos = (head[0] + direction[0], head[1] + direction[1])
            danger = 1 if (not self._in_bounds(next_pos) or next_pos in self.snake) else 0
            vector.append(danger)
        
        # food direction (closer = bigger, normalized 0->1)
        if self.food:
            food_x, food_y = self.food
            head_x, head_y = head

            for direction in directions:
                dx_dir, dy_dir = direction
                # vector from head to food
                fx = food_x - head_x
                fy = food_y - head_y

                # project food vector onto this direction
                dot = fx * dx_dir + fy * dy_dir

                if dot > 0:
                    # distance along this direction
                    distance_along = dot
                    # normalize so that closer = bigger input
                    normalized = 1.0 - min(distance_along / (self.grid_size - 1), 1.0)
                    normalized = max(normalized, 0.0)  # clamp
                else:
                    normalized = 0.0

                vector.append(normalized)
        else:
            # no food on board
            vector.extend([0.0, 0.0, 0.0])
        
        
        # distance to walls (normalized to 0->1)
        for direction in directions:
            # find distance in this direction until hitting wall
            distance = 0
            pos = head
            while self._in_bounds(pos):
                pos = (pos[0] + direction[0], pos[1] + direction[1])
                distance += 1
            # normalize by max distance
            normalized = distance / max(self.grid_size, 1)
            vector.append(normalized)
        
        return vector


    def _turn_relative(self, action: int):
        # action: 0 straight, 1 left, 2 right
        dx, dy = self.direction
        if action == 0:
            return dx, dy
        # left
        if action == 1:
            return -dy, dx
        # right
        return dy, -dx

    def _in_bounds(self, pos: Cell) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def step(self, action: int = 0):
        if self.done:
            return 0, True

        # update direction
        self.direction = self._turn_relative(action)

        head_x, head_y = self.snake[-1]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # collision
        if (not self._in_bounds(new_head)):
            self.done = True
            return -20, True
        if new_head in self.snake:
            self.done = True
            return -100, True

        # move
        self.snake.append(new_head)
        if new_head == self.food:
            self.score += 1
            reward = 30
            self.spawn_food()
        else:
            self.snake.popleft()
            reward = 0

        self.frame += 1

        if self.render_enabled:
            self._draw()
            self._clock.tick(self.fps)

        return reward, False

    def _draw(self):
        screen = self._screen
        cell = self.cell_size
        # colors
        bg = (18, 18, 18)
        gridc = (28, 28, 28)
        snake_head_c = (0, 200, 0)
        snake_body_c = (0, 120, 0)
        food_c = (200, 0, 0)

        screen.fill(bg)
        # grid
        for x in range(0, self.width, cell):
            pygame.draw.line(screen, gridc, (x, 0), (x, self.height))
        for y in range(0, self.height, cell):
            pygame.draw.line(screen, gridc, (0, y), (self.width, y))

        # draw food
        if self.food:
            fx, fy = self.food
            pygame.draw.rect(screen, food_c, (fx * cell, fy * cell, cell, cell))

        # draw snake
        for i, (sx, sy) in enumerate(self.snake):
            color = snake_body_c
            if i == len(self.snake) - 1:
                color = snake_head_c
            pygame.draw.rect(screen, color, (sx * cell, sy * cell, cell, cell))

        # score
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Score: {self.score}", True, (200, 200, 200))
        screen.blit(text, (5, 5))

        pygame.display.flip()


def run_interactive(fps=10):
    game = Game(render=True, fps=fps)

    dir_map = {
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_w: (0, -1),
        pygame.K_s: (0, 1),
        pygame.K_a: (-1, 0),
        pygame.K_d: (1, 0),
    }

    while True:
        print(game.get_input_vector())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_r:
                    game.reset()
                if event.key in dir_map:
                    # set absolute direction (keyboard control)
                    game.direction = dir_map[event.key]

        # step with straight action (keyboard moves direction directly)
        reward, done = game.step(0)
        if done:
            pygame.time.wait(600)
            game.reset()

def run_ga(from_saved: str = None, to_save: str = "vx/best_gen_{}.npy", max_generations: int = np.inf):

    individuals_per_generation = 20

    generation = [] # networks in current generation
    
    if from_saved:
        if not os.path.exists(from_saved):
            print(f"No saved genome found at {from_saved}")
            return

        genome = np.load(from_saved)
        net = nn.NeuralNetworkFixed(input_size=9, hidden_size=6, output_size=3)
        net.set_from_genome(genome)
        generation.append(net)

    while len(generation) < individuals_per_generation:
        generation.append(nn.NeuralNetworkFixed(input_size=9, hidden_size=6, output_size=3))

    generation_num = 0

    while True:
        
        fitness = []
        seed = generation_num # use generation number as seed

        # GENERATION LOOP
        for individual in range(individuals_per_generation):

            net = generation[individual]
            total_reward = 0
            step = 0

            visited = []

            game = Game(render=True, fps=500, seed=seed) # create new game instance for each individual with same seed
            

            # GAME LOOP
            while True:
                
                input_vec = game.get_input_vector()

                output = net.forward(input_vec)

                action = output.argmax() # choose action with highest output

                reward, done = game.step(action)
                step += 1
                total_reward += reward

                x_head, y_head = game.snake[-1]

                # penalize for visiting recent cells to encourage exploration
                if (x_head, y_head) in visited:
                    total_reward -= .5
                else:
                    visited.append((x_head, y_head))

                # prune visited to latest 10
                if len(visited) > 10:
                    visited.pop()

                # give reward for going towards food
                if game.food:
                    food_x, food_y = game.food
                    current_distance = abs(x_head - food_x) + abs(y_head - food_y)
                    if len(game.snake) > 1:
                        prev_x, prev_y = game.snake[-2]
                        prev_distance = abs(prev_x - food_x) + abs(prev_y - food_y)
                        if current_distance < prev_distance:
                            total_reward += 0.2

                # end run if done or too long
                if done or step > 400:
                    pygame.time.wait(600)
                    game.reset()
                    break

            fitness.append(total_reward)
            print(f"Individual {individual} fitness: {total_reward}")
            

        print(f"Generation {generation_num} complete.\n-------------")

        generation_num += 1
        
        # SELECTION
        selected = sorted(zip(generation, fitness), key=lambda x: x[1], reverse=True)[:individuals_per_generation//2]
        selected_nets = [s[0] for s in selected]

        # save best-performing network if it improves
        best_net = selected_nets[0]
        best_fit = fitness[fitness.index(max(fitness))]
        genome = best_net.to_genome()
        best_genome_file = to_save.format(generation_num)
        os.makedirs(os.path.dirname(best_genome_file), exist_ok=True)
        np.save(best_genome_file, genome)
        print(f"(Best fitness {best_fit}. Saved genome to {best_genome_file})")

        # stop if we reach max generations
        if generation_num >= max_generations:
            print("Reached max generations. Stopping.")
            break

        # CROSSOVER
        new_generation = []

        # keep top 2 performers unchanged (elitism)
        top_elite = selected_nets[:2]
        new_generation.extend(top_elite)

        # generate the rest via crossover
        while len(new_generation) < individuals_per_generation:
            parent1 = random.choice(selected_nets)
            parent2 = random.choice(selected_nets)
            
            child = nn.NeuralNetworkFixed(input_size=9, hidden_size=6, output_size=3)

            # simple crossover: half genome from each parent
            genome1 = parent1.to_genome()
            genome2 = parent2.to_genome()
            crossover_point = len(genome1) // 2
            child_genome = np.concatenate([genome1[:crossover_point], genome2[crossover_point:]])
            child.set_from_genome(child_genome)

            new_generation.append(child)

        generation = new_generation

        # MUTATION (except 2)
        for net in generation[2:]:
            net.mutate(mutation_rate=0.1, mutation_strength=0.5)

        print(f"Starting generation {generation_num}...")

def run_save(saved_genome: str):
    best_genome_file = saved_genome
    if not os.path.exists(best_genome_file):
        print(f"No saved genome found at {best_genome_file}")
        return

    genome = np.load(best_genome_file)
    net = nn.NeuralNetworkFixed(input_size=9, hidden_size=6, output_size=3)
    net.set_from_genome(genome)

    game = Game(render=True, fps=20, seed=42)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        input_vec = game.get_input_vector()
        output = net.forward(input_vec)
        action = output.argmax()
        reward, done = game.step(action)
        if done:
            pygame.time.wait(1000)
            game.reset()

if __name__ == '__main__':
    run_save("v4/best_gen_1000.npy")
    # run_ga(from_saved="v3/best_gen_2000.npy", to_save="v4/best_gen_{}.npy", max_generations=1000)
