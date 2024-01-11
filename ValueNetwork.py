import imports_file
class ValueNetwork(nn.Module):
    def __init__(self,inputShape,lr=0.0005):
        super(ValueNetwork,self).__init__()
        self.l1=nn.Linear(*inputShape,256)
        self.l2=nn.Linear(256,256)
        self.l3=nn.Linear(256,1)
        self.optimizer=optim.Adam(self.parameters(),lr)
    def forward(self,input):
        output=F.relu(self.l1(input)) 
        output=F.relu(self.l2(output))
        return self.l3(output)
