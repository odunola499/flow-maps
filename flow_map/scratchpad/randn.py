import torch

def calc_diff():
    B = 2
    N = 10
    temp1 = torch.rand([B * N])
    temp2 = torch.rand([B * N])
    s = torch.minimum(temp1, temp2)
    t = torch.maximum(temp1, temp2)
    t = t-s
    print(torch.nn.functional.mse_loss(s, t))

if __name__ == '__main__':
    calc_diff()