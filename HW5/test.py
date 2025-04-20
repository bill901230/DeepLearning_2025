import torch
sd = torch.load("./results/pong/best_model.pt", map_location="cpu")
first_w = sd["network.0.weight"]
print(first_w.shape) 