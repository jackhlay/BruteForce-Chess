# Standard library imports
import asyncio
import os
import random
import logging
import pprint
from queue import Queue
import time
from typing import List


# Third-party imports
from fastapi import FastAPI, HTTPException
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
import redis
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
epochs = 29
cycles = 0
batch_size = 1024
epsilon = .9  # Exploration rate
lr = 1e-5
gamma = 0.9
epsilon_decay = 0.995
min_epsilon = 0.001

# Constants
board_depth = 12
board_size = 8

# Action map to convert string actions to integer indices
action_map = {}
action_size = len(action_map)   #action size
validationLosses = []
global model, idles, totalIdles

# Global Data queue for storing samples (shared across FastAPI app and training loop)
data_queue = Queue(maxsize=750)


# FastAPI for receiving data
app = FastAPI()

class PosData(BaseModel):
    start_fen: str
    start_rating: float
    action: str
    end_fen: str
    end_rating: float

#Redis Connection
red = redis.Redis(
    host = "localhost",
    port = 6379
    )

# Define the model
class theNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(theNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3) # Dropout layer to prevent overfitting

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer without activation
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
        return torch.from_numpy(tensor).flatten()
    except Exception as e:
        raise ValueError(f"Invalid FEN string: {e}")

# Function to get the action index, adding it if it's not already in the map
def get_action_index(action):
    if action not in action_map:
        # Assign a new index for the new action
        action_map[action] = len(action_map)
    return action_map[action]

# Training loop:
def train(training_data: List[dict]):
    global epsilon, cycles, model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        random.shuffle(training_data)
        batch_loss = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            # Prepare tensors
            states = torch.stack([data["start_tensor"].flatten() for data in batch])
            next_states = torch.stack([data["end_tensor"].flatten() for data in batch])
            rewards = torch.tensor([data["end_rating"] for data in batch], dtype=torch.float32)  # Use the end rating directly
            actions = torch.tensor([get_action_index(data["action"]) for data in batch], dtype=torch.long)  # Action index (if used)

            # Forward pass: Predict the next state rating directly
            predicted_ratings = model(states)

            # Loss: Use the rating difference between predicted and actual ratings
            loss = criterion(predicted_ratings.squeeze(), rewards)  # Loss based on predicted vs. actual end ratings
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()


        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {batch_loss:.4f}, Epsilon: {epsilon:.4f}")
        validationLosses.append(batch_loss)
        cycles += 1

# Background training loop
async def training_loop(red: redis.Redis):

    global model, idles, totalIdles, cycles
    logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    idles, totalIdles = 0, 0
    global epsilon

    while True:
        # if idles > 1040:
        #     logging.info("Queue Idle for too long. Saving weights and exiting...")
        #     break

        # queue_size = data_queue.qsize()  # Capture the queue size at the start of the loop
        queue_size = 0
        print(f"REDIS SET SIZE:{queue_size}")
        if queue_size >= 1000000:
            # Get a batch of data for training
            training_data = list([data_queue.get() for _ in range(batch_size)])
            train(training_data)
            logging.info("Batch processed. Waiting for more data...")
            idles = 0
        else:
            # Print and log the current queue size continuously
            logging.info(f"Queue size: {queue_size}. Idles: {idles}. Total Idles: {totalIdles}. Waiting for more data...")

        time.sleep(1.3)  # Sleep for a short time before rechecking the queue size
        if queue_size == data_queue.qsize():
            idles += 1
            totalIdles += 1
    # cleanup(start_time)

def cleanup(start_time):
    global validationLosses, model
    pprint.pprint(f"WEIGHTS / STATE DICT: {model.state_dict()}")

    logging.info(f"Model Trained for a total of {time.time() - start_time} seconds")
    torch.save(model.state_dict(), "model_weights.txt")  # Save model's state_dict
    plt.plot(np.arange(len(validationLosses)), validationLosses)  # Plot the actual values with indices as the x-axis

    # Customize the plot
    plt.title('Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Save the plot to a file
    plt.savefig(f"lossgraph_{time.time()}.png")  # Use a timestamp to generate unique filenames

async def start_training():
    global model
    red = redis.Redis(host=os.getenv("REDHOST", "localhost"), port=os.getenv("REDPORT", 6379), db=1)
    model = theNN(768, 512, batch_size)
    asyncio.create_task(training_loop(red)) # Start the training loop in the background

if __name__ == "__main__":
    print("starting")
    asyncio.run(start_training())
