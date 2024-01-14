class PolicyNetwork(nn.Module):
    def __init__(self,inputShape,outputShape,lr):
        super(PolicyNetwork,self).__init__()
        self.l1=nn.Linear(*inputShape,128)
        self.l2=nn.Linear(128,128)
        self.l3=nn.Linear(128,outputShape)
        self.optimizer=optim.Adam(self.parameters(),lr)
    def forward(self,input):
        output=F.relu(self.l1(input)) 
        output=F.relu(self.l2(output))
        return self.l3(output)