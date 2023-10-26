import numpy as np 
from tqdm import tqdm
class RNN:
    def __init__(self, x, y, hidden_units):
        self.x = x # shape [samples, timesteps, features]
        self.y = y # shape [samples, outputs]
        self.hidden_units = hidden_units
        self.Wx = np.random.randn(self.hidden_units, self.x.shape[2])
        self.Wh = np.random.randn(self.hidden_units, self.hidden_units)
        self.Wy = np.random.randn(self.y.shape[1],self.hidden_units)
    
    def cell(self, xt, ht_1):
        ht = np.tanh(np.dot(self.Wx,xt.reshape(1,1)) + np.dot(self.Wh,ht_1))
        yt = np.dot(self.Wy,ht)
        return ht, yt
        
    def forward(self, sample):
        sample_x, sample_y = self.x[sample], self.y[sample]
        ht = np.zeros((self.hidden_units,1)) # first hidden state is zeros vector
        self.hidden_states = [ht] # collection of hidden states for each sample
        self.inputs = [] # collection of inputs for each sample
        for step in range(len(sample_x)):
            ht, yt = self.cell(sample_x[step],ht)
            self.inputs.append(sample_x[step].reshape(1,1))
            self.hidden_states.append(ht)
        self.error = yt - sample_y
        self.loss = 0.5*self.error**2
        self.yt = yt

    def backward(self):
        n = len(self.inputs)
        dyt = self.error # dL/dyt
        dWy = np.dot(dyt,self.hidden_states[-1].T) # dyt/dWy
        dht = np.dot(dyt, self.Wy).T # dL/dht = dL/dyt * dyt/dht ,where ht = tanh(Wx*xt + Wh*ht))
        dWx = np.zeros(self.Wx.shape)
        dWh = np.zeros(self.Wh.shape)
        # BPTT
        for step in reversed(range(n)):
            temp = (1-self.hidden_states[step+1]**2) * dht # dL/dtanh = dL/dyt * dyt/dht * dht/dtanh, where dtanh = (1-ht**2) 
            dWx += np.dot(temp, self.inputs[step].T) # dL/dWx = dL/dyt * dyt/dht * dht/dtanh * dtanh/dWx
            dWh += np.dot(temp, self.hidden_states[step].T) # dL/dWh = dL/dyt * dyt/dht * dht/dtanh * dtanh/dWh

            dht = np.dot(self.Wh, temp) # dL/dht-1 = dL/dht * (1 - ht+1^2) * Whh
        dWy = np.clip(dWy, -1, 1)
        dWx = np.clip(dWx, -1, 1)
        dWh = np.clip(dWh, -1, 1)
        self.Wy -= self.lr * dWy
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        
    def train(self, epochs, learning_rate):
        self.Ovr_loss = []
        self.lr = learning_rate
        for epoch in tqdm(range(epochs)):
            for sample in range(self.x.shape[0]):
                self.forward(sample)
                self.backward()
            self.Ovr_loss.append(np.squeeze(self.loss / self.x.shape[0]))
            self.loss = 0     
        
    def predict(self,x,y):
        outputs = []
        for sample in range(len(x)):
            self.forward(sample)
            outputs.append(self.yt)
        return np.array(outputs).reshape(y.shape)

def sin_dataset_generator(size =  200, timesteps = 25, phase = 1):
    '''
    Parameters:
        size: The length of the sine wave. By default, it's set to 200.
        timesteps: The number of steps in each input sequence. By default, it's set to 25.
        phase: The phase shift for the sine wave. By default, it's set to 1.
    
    Functionality:
        The function first generates a sine wave of length size with a phase shift of phase using np.sin().
        It then creates input sequences (x) and their corresponding next-step targets (y) from the sine wave.
        For each position in the sine wave (except the last timesteps positions), it takes the next timesteps values as an input sequence and the value right after those timesteps as the target.
        This way, given a sequence of timesteps sine wave values, the model will be trained to predict the next value in the sine wave.
    
    Returns:
        x: A numpy array of shape (number_of_samples, timesteps, 1). Each sample is a sequence of timesteps sine wave values.
        y: A numpy array of shape (number_of_samples, 1). Each value in y is the next sine wave value following the corresponding sequence in x.
    '''
    x, y = [], []
    sin_wave = np.sin(np.arange(0,size,phase))
    for step in range(sin_wave.shape[0]-timesteps):
        x.append(sin_wave[step:step+timesteps])
        y.append(sin_wave[step+timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)