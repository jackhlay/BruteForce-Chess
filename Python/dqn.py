import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging
import os
import threading
from queue import Queue
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

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

# Data queue for storing samples
data_queue = Queue(maxsize=7000)

# FastAPI for receiving data
app = FastAPI()


class PosData(BaseModel):
    start_fen: str
    start_rating: float
    action: str
    end_fen: str
    end_rating: float


# FastAPI route
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

        data_queue.put(sample)
        return {"message": "Position added to queue", "queue_size": data_queue.qsize()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process position: {e}")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(board_depth, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


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


def train(model: nn.Module, optimizer, criterion, training_data: List[dict]):
    global epsilon
    for epoch in range(epochs):
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        random.shuffle(training_data)
        batch_loss = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            states = torch.stack([torch.tensor(data["start_tensor"]) for data in batch])
            next_states = torch.stack([torch.tensor(data["end_tensor"]) for data in batch])
            rewards = torch.tensor([data["end_rating"] - data["start_rating"] for data in batch], dtype=torch.float32)
            actions = torch.tensor([data["action"] for data in batch], dtype=torch.long)

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


def training_work():
    logging.basicConfig(level=logging.INFO)
    model = DQN()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    while 1==1:
        if data_queue.qsize() >= 1100:
            training_data = [data_queue.get() for _ in range(1100)]
            train(model, optimizer, criterion, training_data)
            logging.info("Batch processed. Waiting for more data...")
        else:
            print(f"Queue size: {data_queue.qsize()}. Waiting for more data...")
            logging.info(f"Queue size: {data_queue.qsize()}. Waiting for more data...")
            time.sleep(2)


if __name__ == "__main__":
    threading.Thread(target=training_work, daemon=True).start()
    training_work()
