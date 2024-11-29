import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import uvicorn
import logging
from queue import Queue
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from asyncio import create_task, sleep

# Hyperparameters
epochs = 100
batch_size = 100
epsilon = 0.1  # Exploration rate
lr = 1e-4
gamma = 0.99
epsilon_decay = 0.99
min_epsilon = 0.01

# Constants
board_depth = 12
board_size = 8
action_size = 300  # Example action size

# Global Data queue for storing samples (shared across FastAPI app and training loop)
data_queue = Queue(maxsize=7000)

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

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8, 768)
        self.fc2 = nn.Linear(768,64)
        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)

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

# Action map to convert string actions to integer indices
action_map = {
    "e2e4": 0, "e2e5": 1, "d2d4": 2,  # example action mappings
    # Add other possible actions here
}

# Function to get the action index, adding it if it's not already in the map
def get_action_index(action):
    if action not in action_map:
        # Assign a new index for the new action
        action_map[action] = len(action_map)
    return action_map[action]

# Training loop:
def train(model: nn.Module, optimizer, criterion, training_data: List[dict]):
    global epsilon
    for epoch in range(epochs):
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        random.shuffle(training_data)
        batch_loss = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            states = torch.stack([data["start_tensor"].clone().detach() for data in batch])
            next_states = torch.stack([data["end_tensor"].clone().detach() for data in batch])
            rewards = torch.tensor([data["end_rating"] - data["start_rating"] for data in batch], dtype=torch.float32)

            # Convert actions from string to integer index, dynamically adding new actions
            actions = torch.tensor([get_action_index(data["action"]) for data in batch], dtype=torch.long)

            q_values = model(states)
            next_q_values = model(next_states)

            target_q_values = rewards + gamma * next_q_values.max(1)[0]
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            loss = criterion(q_value, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        logging.info(f"Epoch {epoch}/{epochs}, Loss: {batch_loss}")


# Background training loop
async def training_loop():
    logging.basicConfig(level=logging.INFO)
    model = DQN()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    while True:
        queue_size = data_queue.qsize()  # Capture the queue size at the start of the loop
        if queue_size >= 300:
            # Get a batch of data for training
            training_data = [data_queue.get() for _ in range(300)]
            train(model, optimizer, criterion, training_data)
            logging.info("Batch processed. Waiting for more data...")
        else:
            # Print and log the current queue size continuously
            print(f"Queue size: {queue_size}. Waiting for more data...")
            logging.info(f"Queue size: {queue_size}. Waiting for more data...")

        await sleep(0.7)  # Sleep for a short time before rechecking the queue size

@app.on_event("startup")
async def start_training():
    # Start the training loop in the background when the FastAPI server starts
    create_task(training_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
