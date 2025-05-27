# %%

import os
from dotenv import load_dotenv
import wandb
import random

# %%
# Load environment variables from .env file
load_dotenv(dotenv_path="C:/Users/idiaz/OneDrive - IE University/00. IE Courses/01. 2025_H1/4. MLOps/My projects/main_logging_examples/WandB_demo/.env")
# %%
wandb.login()  # Login to WandB using the API key from the environment variable
# wandb.login(key=os.getenv("WANDB_API_KEY"))

# %%
# set up experiement
epochs = 10
lr = 0.01

# %%

run = wandb.init(
    # Setting up the WandB project
    project="wandb_basic_demo",
    # Track hyperparameters and run metadata
    config={
        "epochs": epochs,
        "learning_rate": lr,
    },
)

# %%
# Simulate training and logging metrics

offset = random.random() / 5
for epoch in range(1, epochs + 1):
    acc = 1 - 2 ** - epoch - random.random() / epoch - offset
    loss = 2 ** - epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, acc={acc:.4f}, loss={loss:.4f}")
    # Log metrics to WandB
    wandb.log({"accuracy": acc, "loss": loss})

# %%
# log the code that was run
run.log_code()

# %%
wandb.finish()  # End the WandB run

# %%
