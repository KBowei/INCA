import torch

def channel2batch(x, channel_in):
    b, c, h, w = x.size()
    assert c == channel_in
    out = torch.zeros(b*c, 1, h, w).to(x.device)
    for i in range(channel_in):
        out[i*b: (i+1)*b]= x[:, i:i+1]
    return out

def batch2channel(x, channel_in):
    b, c, h, w = x.size()
    batch = b//channel_in
    assert c == 1
    out = torch.zeros(batch, channel_in, h, w).to(x.device) 
    for i in range(channel_in):     
        out[:, i:i+1] = x[i*batch: (i+1)*batch]
    return out  