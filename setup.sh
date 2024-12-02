#!/bin/bash
# This script sets up the project by installing dependencies, checking for a poetry environment,
# and installing pre-commit hooks.

# Add color and formatting variables at the top
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function for section headers
print_section() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}\n"
}

# Function for success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function for warnings
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Initialize and update git submodules
print_section "Git Submodules"
echo "Initializing git submodules..."
git submodule update --init --recursive

# ---- ENVIRONMENT VARIABLES ---- #
# Source .env file if it exists
print_section "Environment Variables"
if [ -f .env ]; then
    print_success "Loading environment variables from .env..."
    source .env
else
    print_warning "No .env file found. You might need to create one with HF_TOKEN and WANDB_API_KEY"
    echo -e "${YELLOW}Example .env contents:${NC}"
    echo "export HF_TOKEN=your_huggingface_token"
    echo "export WANDB_API_KEY=your_wandb_key"
fi

# ---- EVALUATION SETUP ---- #
# Clone Paloma dataset if credentials are provided and directory doesn't exist
print_section "Evaluation Setup"
if [ ! -d "lib/paloma" ]; then
    if [ ! -z "$HF_TOKEN" ]; then
        echo "Cloning Paloma evaluation dataset..."
        git clone https://oauth2:${HF_TOKEN}@huggingface.co/datasets/allenai/paloma lib/paloma
        print_success "Paloma dataset cloned successfully"
    else
        print_warning "Skipping Paloma dataset clone. To clone, provide HuggingFace credentials"
    fi
else
    print_success "Paloma dataset already exists, skipping clone"
fi

# Create environment for running evaluation inside of lib/olmo_eval
# skip if already exists
if [ ! -d "lib/olmo-eval/env" ]; then
    print_section "OLMo Eval Setup"
    cd lib/olmo-eval
    echo "Creating virtual environment..."
    virtualenv env
    source env/bin/activate
    pip install -e .
    deactivate
    cd ../../
    print_success "OLMo eval environment setup complete"
else
    print_success "olmo-eval environment already exists, skipping setup"
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

print_section "Poetry Environment Setup"
if ! check_poetry_env; then
    echo "No poetry environment found. Initializing..."
    poetry install --with dev --no-root
    print_success "Poetry environment created successfully"
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

