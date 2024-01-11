class ReplayBuffer:
    def __init__(self, maxSize, stateDim):
        self.state = np.zeros((maxSize, stateDim))
        self.action = np.zeros(maxSize, dtype=np.int8)
        self.reward = np.zeros(maxSize)
        self.done = np.zeros(maxSize, dtype=np.int8)
        self.nextState = np.zeros((maxSize, stateDim))
        self.maxSize = maxSize
        self.curser = 0
        self.size = 0

    def save(self, state, action, reward, nextState, done):
        self.state[self.curser] = state
        self.action[self.curser] = action
        self.reward[self.curser] = reward
        self.nextState[self.curser] = nextState
        self.done[self.curser] = done
        self.curser = (self.curser + 1) % self.maxSize
        if self.size < self.maxSize:
            self.size += 1

    def sample(self, batchSize):
        batchSize = min(self.size, batchSize - 1)
        indexes = np.random.choice([i for i in range(self.size - 1)], batchSize)
        return self.state[indexes], self.action[indexes], self.reward[indexes], self.nextState[indexes], self.done[
            indexes]
