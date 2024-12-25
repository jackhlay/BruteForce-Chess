# Standard library imports
import random
import logging
import pprint
from queue import Queue
from typing import List
from asyncio import create_task, sleep
 
# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Hyperparameters
epochs = 20
batch_size = 128
epsilon = 0.1  # Exploration rate
lr = 1e-4
gamma = 0.99
epsilon_decay = 0.995
min_epsilon = 0.01

# Constants
board_depth = 12
board_size = 8

# Action map to convert string actions to integer indices
action_map = {}

action_size = len(action_map)   #action size


# Global Data queue for storing samples (shared across FastAPI app and training loop)
data_queue = Queue(maxsize=16384)

# FastAPI for receiving data
app = FastAPI()

class PosData(BaseModel):
    start_fen: str
    start_rating: float
    action: str
    end_fen: str
    end_rating: float

# FastAPI route to add data to the queue
@app.post("/")
async def add_pos(pos: PosData):
    if data_queue.full():
        raise HTTPException(status_code=503, detail="Queue is full. Try again later.")

    try:
        pprint.pprint(pos)
        start_tensor = fen_to_tensor(pos.start_fen)
        end_tensor = fen_to_tensor(pos.end_fen)

        sample = {
            "start_tensor": start_tensor,
            "start_rating": pos.start_rating,
            "action": pos.action,
            "end_tensor": end_tensor,
            "end_rating": pos.end_rating,
        }

        # Put sample in the shared queue
        data_queue.put(sample)
        return {"message": "Position added to queue", "queue_size": data_queue.qsize()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process position: {e}")


# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)  # Input: 12 channels
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer to reduce spatial size
        
        # The input size of the following linear layer depends on the input image size
        # Assuming the input image size is 64x64, we compute the output size after conv and pooling
        # After 3 convolution layers and pooling, the image size is reduced by a factor of 2 each time (assuming kernel=2 for pooling)
        self.fc1 = nn.Linear(1024, 512)  # Flattened size after convolution and pooling
        
        self.fc2 = nn.Linear(256, batch_size)  # Output layer: one score per position

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # Pool after first conv layer
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # Pool after second conv layer
        x = torch.relu(self.conv3(x))
        x = self.pool(x)  # Pool after third conv layer
        
        x = x.view(x.size(0), -1)  # Flatten the tensor before feeding into the fully connected layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Output layer
        return x
    
def fen_to_tensor(fen: str) -> torch.Tensor:
    piece_map = {
        'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,
        'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12
    }
    tensor = np.zeros((board_depth, board_size, board_size), dtype=np.float32)

    try:
        board = fen.split()[0].split('/')
        for i, row in enumerate(board):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    piece_id = piece_map.get(char, 0)
                    tensor[piece_id - 1, i, col_idx] = 1
                    col_idx += 1
        return torch.from_numpy(tensor)
    except Exception as e:
        raise ValueError(f"Invalid FEN string: {e}")

# Function to get the action index, adding it if it's not already in the map
def get_action_index(action):
    if action not in action_map:
        # Assign a new index for the new action
        action_map[action] = len(action_map)
    return action_map[action]

# Training loop:
def train(model: nn.Module, optimizer, criterion, training_data: List[dict]):
    global epsilon
    model.train()  # Set the model to training mode

    best_loss = float('inf')

    for epoch in range(epochs):
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        random.shuffle(training_data)
        batch_loss = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            if len(batch) == 0:
                print("Skipping empty batch")
                continue  # Skip empty batches

            states = torch.stack([data["start_tensor"].clone().detach() for data in batch])
            next_states = torch.stack([data["end_tensor"].clone().detach() for data in batch])
            rewards = torch.tensor([data["end_rating"] - data["start_rating"] for data in batch], dtype=torch.float32)

            # Convert actions from string to integer index, dynamically adding new actions
            actions = torch.tensor([get_action_index(data["action"]) for data in batch], dtype=torch.long)

            q_values = model(states)
            next_q_values = model(next_states)

            if next_q_values.size(1) == 0:
                continue  # Skip if next_q_values is empty

            target_q_values = rewards + gamma * next_q_values.max(1)[0].detach()
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            loss = criterion(q_value, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {batch_loss:.4f}, Epsilon: {epsilon:.4f}")

# Background training loop
async def training_loop():
    idles = 0
    logging.basicConfig(level=logging.INFO)
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    global epsilon

    while True:
        queue_size = data_queue.qsize()  # Capture the queue size at the start of the loop
        if queue_size >= batch_size:
            # Get a batch of data for training
            training_data = [data_queue.get() for _ in range(batch_size)]
            train(model, optimizer, criterion, training_data)
            logging.info("Batch processed. Waiting for more data...")
        else:
            # Print and log the current queue size continuously
            logging.info(f"Queue size: {queue_size}. Idles: {idles}. Waiting for more data...")

        await sleep(1.73)  # Sleep for a short time before rechecking the queue size
        if queue_size == data_queue.qsize():
            idles += 1

@app.on_event("startup")
async def start_training():
    create_task(training_loop()) # Start the training loop in the background

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
