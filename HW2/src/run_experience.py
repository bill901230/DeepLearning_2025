import os

data_path = "./dataset/oxford-iiit-pet"  
epochs = 50
learning_rate = 1e-3
model_name = "unet"
# model_name = "resnet34"
# gpu = "0"
gpu = "2"

batch_sizes = [4]

for bs in batch_sizes:
    print(f"Running {model_name} with batch_size {bs} on GPU {gpu}")
    command = f"CUDA_VISIBLE_DEVICES={gpu} python train.py --data_path {data_path} --epochs {epochs} --batch_size {bs} --learning_rate {learning_rate} --model_name {model_name}"
    print(f"start...")
    os.system(command)
