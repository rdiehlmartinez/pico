#!/bin/bash
# This script sets up the project by installing dependencies, checking for a poetry environment,
# and installing pre-commit hooks.

# Initialize and update git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# ---- ENVIRONMENT VARIABLES ---- #
# Source .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    source .env
else
    echo "Warning: No .env file found. You might need to create one with HF_TOKEN and WANDB_API_KEY"
    echo "Example .env contents:"
    echo "export HF_TOKEN=your_huggingface_token"
    echo "export WANDB_API_KEY=your_wandb_key"
fi

# ---- EVALUATION SETUP ---- #
# Clone Paloma dataset if credentials are provided and directory doesn't exist
if [ ! -d "lib/paloma" ]; then
    if [ ! -z "$HF_TOKEN" ]; then
        echo "Cloning Paloma evaluation dataset..."
        git clone https://oauth2:${HF_TOKEN}@huggingface.co/datasets/allenai/paloma lib/paloma
    else
        echo "Skipping Paloma dataset clone. To clone, provide HuggingFace credentials"
    fi
else
    echo "Paloma dataset already exists, skipping clone"
fi

# Create environment for running evaluation inside of lib/olmo_eval
# skip if already exists
if [ ! -d "lib/olmo-eval/env" ]; then
    cd lib/olmo-eval
    virtualenv env
    source env/bin/activate
    pip install -e .
    deactivate
    cd ../../
else
    echo "olmo-eval environment already exists, skipping setup"
fi

# ---- POETRY ENVIRONMENT SETUP ---- #
# Function to check if poetry environment exists
check_poetry_env() {
    if poetry env list &> /dev/null; then
        return 0  # Environment exists
    else
        return 1  # No environment found
    fi
}

# Check if poetry environment exists
if ! check_poetry_env; then
    echo "No poetry environment found. Initializing..."
    poetry install --with dev --no-root
fi

# Check if we're already in a Poetry shell
if [ -z "$POETRY_ACTIVE" ]; then
    echo "Activating poetry shell..."
    # Execute the remaining commands in the poetry shell
    poetry run bash -c '
        # Install git-lfs
        if ! command -v git-lfs &> /dev/null; then
            echo "Installing git-lfs..."
            git-lfs install
        else
            echo "git-lfs already installed, skipping installation"
        fi

        # Install pre-commit hooks
        echo "Setting up pre-commit hooks..."
        pre-commit install
    '
    poetry shell
else
    # Already in poetry shell, execute commands directly
    # Install git-lfs
    if ! command -v git-lfs &> /dev/null; then
        echo "Installing git-lfs..."
        git-lfs install
    else
        echo "git-lfs already installed, skipping installation"
    fi

    # Install pre-commit hooks
    echo "Setting up pre-commit hooks..."
    pre-commit install
fi

