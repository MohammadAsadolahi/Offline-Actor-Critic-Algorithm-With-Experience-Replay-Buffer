class Agent():
    def __init__(self, inputShape, outputShape, gamma=0.99, lr=5e-3):
        self.policyNet = PolicyNet(inputShape, outputShape, lr)
        self.valueNet = ValueNet(inputShape, lr)
        self.memory = replayBuffer(1000000, env.observation_space.shape[0])
        self.gamma = T.tensor(gamma, dtype=T.float)

    def save(self, state, action, reward, state_, done):
        self.memory.save(state, action, reward, state_, done)
        state = T.tensor([state], dtype=T.float)
        state_ = T.tensor([state_], dtype=T.float)
        with T.no_grad():
            v_ = self.valueNet(state_).detach()
        self.valueNet.optimizer.zero_grad()
        G = (reward + self.gamma * ((1 - done) * v_)) - self.valueNet(state)
        valueLoss = G ** 2
        valueLoss.backward()
        self.valueNet.optimizer.step()

    def chooseAction(self, state):
        state = T.tensor([state], dtype=T.float)
        with T.no_grad():
            probs = F.softmax(self.policyNet.forward(state))
        actionProbs = T.distributions.Categorical(probs)
        action = actionProbs.sample()
        return action.item()

    def learn(self, batchSize):
        if self.memory.size > batchSize:
            self.policyNet.optimizer.zero_grad()
            self.valueNet.optimizer.zero_grad()

            state, action, reward, state_, done = self.memory.sample(batchSize)
            state = T.tensor(state, dtype=T.float)
            state_ = T.tensor(state_, dtype=T.float)
            reward = T.unsqueeze(T.tensor(reward, dtype=T.float), axis=1)
            action = T.tensor(action, dtype=T.float)
            done = T.unsqueeze(T.tensor(done, dtype=T.float), axis=1)

            with T.no_grad():
                v_ = self.valueNet(state_).detach()
            G = (reward + self.gamma * ((1 - done) * v_)) - self.valueNet(state)
            valueLoss = T.mean(G ** 2)
            valueLoss.backward()
            self.valueNet.optimizer.step()
            probs = F.softmax(self.policyNet.forward(state))
            actionProbs = T.distributions.Categorical(probs)

            policyLoss=T.mean(T.squeeze(-actionProbs.log_prob(action) * T.squeeze(G.detach())))
            policyLoss.backward()
            self.policyNet.optimizer.step()
