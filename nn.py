import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
class NeuralNetworkFixed:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.rand(hidden_size, input_size)
        self.b1 = np.random.rand(hidden_size)
        self.w2 = np.random.rand(output_size, hidden_size)
        self.b2 = np.random.rand(output_size)
    
    def forward(self, x):
        hidden = sigmoid(np.dot(self.w1, x) + self.b1)
        output = sigmoid(np.dot(self.w2, hidden) + self.b2)
        return output
    
    def set_from_genome(self, genome):
        idx = 0
        w1_size = self.w1.size
        b1_size = self.b1.size
        w2_size = self.w2.size
        b2_size = self.b2.size
        
        self.w1 = genome[idx:idx+w1_size].reshape(self.w1.shape)
        idx += w1_size
        self.b1 = genome[idx:idx+b1_size]
        idx += b1_size
        self.w2 = genome[idx:idx+w2_size].reshape(self.w2.shape)
        idx += w2_size
        self.b2 = genome[idx:idx+b2_size]
    
    def to_genome(self):
        return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        # convert NN to genome
        genome = self.to_genome()
        
        # mutate each gene with probability mutation_rate
        for i in range(len(genome)):
            if np.random.rand() < mutation_rate:
                genome[i] += np.random.uniform(-mutation_strength, mutation_strength)
                genome[i] = np.clip(genome[i], -1, 1) # clip weights to [-1, 1]
        
        # set the mutated genome back to the NN
        self.set_from_genome(genome)

if __name__ == "__main__":
    nn = NeuralNetworkFixed(input_size=9, hidden_size=6, output_size=3)
    inputs = np.random.rand(9)
    print(nn.forward(inputs))
    genome = nn.to_genome()

    nn2 = NeuralNetworkFixed(input_size=9, hidden_size=6, output_size=3)
    nn2.set_from_genome(genome)

    print(nn2.forward(inputs))