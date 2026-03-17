# Darwin's Snake
Darwin's Snake is a Snake AI that learns to play using neuroevolution.
A neural network controls the snake, and a genetic algorithm evolves its parameters over generations.

Here's a great [Medium article on genetic algorithms](https://medium.com/@bsaladkari/real-world-applications-of-genetic-algorithms-7b223125e2b7) that I referenced while building this project.

## How It Works

The snake is controlled by a small neural network:

Inputs (9): danger straight, danger left, danger right, food distance straight, food distance left, food distance right, wall distance straight, wall distance left, wall distance right.

Outputs (3): 0 = straight, 1 = turn left, 2 = turn right

*Directions are relative to the direction the snake is moving.*

### Genetic Algorithm

Each generation:
1. Evaluate all snakes' fitness
2. Select best performers
3. Preserve top elite networks
4. Create children via crossover
6. Mutate weights randomly
7. Repeat

### Fitness
**Rewards**: eating food, moving towards food

**Penalties**: dying (wall), dying (self), visiting recent cells


## My Training Process

I first trained it with extra rewards to incentivize exploration: moving closer to food gives reward and visiting recently visited cell gives penalty.
Once it gets good at that, I will remove the extra rewards, leaving only food and death, in order to maximize efficiency.


(9, 6, 3)
- v1
- v2
- v3: added higher penalty for suicide, 2000 generations
- v4 (v3): +1000 generations, higher step limit (200 -> 400)
- v5: new crossover algorithm, step limit = 750
    - avoids suicide by taking wide paths to food, very inefficient
- v6 (v5): +20000 generations, step limit = 1000
(10, 6, 3)
- v7: snake length as input
(10, 16, 3)
- v8
- v9 (v8): removed incentives for exploration
- v10 (v9): step limit = 3000, no incentives
