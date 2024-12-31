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

# Initialize error tracking
ERRORS_FOUND=0

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

# --- GIT LFS SETUP --- #
print_section "Git LFS Setup"
if ! command -v git-lfs &> /dev/null; then
    print_warning "git-lfs is not installed. Some model checkpointing functionality may not work correctly."
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
    
    # Check the operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo -e "${YELLOW}    You can install it using Homebrew:${NC}"
        echo "    brew install git-lfs"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo -e "${YELLOW}    You can install it using your package manager:${NC}"
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            echo "    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash"
            echo "    sudo apt-get install git-lfs"
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            echo "    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash"
            echo "    sudo yum install git-lfs"
        else
            print_warning "Could not detect package manager. Please install git-lfs manually."
        fi
    else
        print_warning "Unsupported operating system. Please install git-lfs manually."
    fi
else
    git-lfs install 
    print_success "git-lfs installed and initialized"
fi

# --- CUDA VERSION CHECK --- #
print_section "CUDA Version Check"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')
    
    if [[ -z "$CUDA_VERSION" ]]; then
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
        print_warning "nvidia-smi failed to communicate with the NVIDIA driver."
        echo -e "${YELLOW}    Ensure that the latest NVIDIA driver is installed and running.${NC}"
    else
        MAJOR_VERSION=${CUDA_VERSION%.*}
        MINOR_VERSION=${CUDA_VERSION#*.}
        
        if [ "$MAJOR_VERSION" -lt 12 ] || ([ "$MAJOR_VERSION" -eq 12 ] && [ "$MINOR_VERSION" -lt 1 ]); then
            ERRORS_FOUND=$((ERRORS_FOUND + 1))
            print_warning "CUDA version ${MAJOR_VERSION}.${MINOR_VERSION} detected."
            echo -e "${YELLOW}    Some multi-node communication GPU features may not work properly.${NC}"
            echo -e "${YELLOW}    CUDA version 12.1 or newer is recommended.${NC}"
        else
            print_success "CUDA version ${MAJOR_VERSION}.${MINOR_VERSION} detected"
        fi
    fi
else
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
    print_warning "nvidia-smi not found. Unable to check CUDA version."
    echo -e "${YELLOW}    Ensure that NVIDIA drivers and CUDA version at 12.1 or newer are installed for GPU support.${NC}"
fi


# ---- ENVIRONMENT VARIABLES ---- #
print_section "Environment Variables"
if [ -f .env ]; then
    print_success "Loading environment variables from .env..."
    source .env
    if [[ -n "$HF_TOKEN" && -n "$WANDB_API_KEY" ]]; then
        print_success "Both HF_TOKEN and WANDB_API_KEY are set and loaded!"
    else
        print_warning "One or both of HF_TOKEN and WANDB_API_KEY are not set."
    fi
else
    print_warning "No .env file found."
    echo -e "${YELLOW}    You might need to create one with HF_TOKEN and WANDB_API_KEY${NC}"
    echo -e "${YELLOW}    Example .env contents:${NC}"
    echo "    export HF_TOKEN=your_huggingface_token"
    echo "    export WANDB_API_KEY=your_wandb_key"
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
fi

# ---- POETRY SETUP ---- #
print_section "Poetry Setup"

# First check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    print_success "Poetry installed successfully"
else
    print_success "Poetry already installed"
fi

# Then check for virtual environment
if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Creating one..."
    poetry config virtualenvs.in-project true
    poetry install --with dev 
    print_success "Poetry environment created successfully"
else
    print_success "Poetry environment already exists"
fi

# ---- PRE-COMMIT SETUP ---- #
print_section "Pre-commit Setup"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
poetry run pre-commit install
print_success "Pre-commit hooks installed"

# Run pre-commit hooks on all files
echo "Running pre-commit hooks on all files..."
poetry run pre-commit run --all-files
print_success "Pre-commit initial run complete"

# --- Final Status Message --- #

# Final status message
print_section "Setup Status"
if [ $ERRORS_FOUND -eq 0 ]; then
    print_success "Setup Complete! 🎉"
    print_success "To activate the virtual environment, run: poetry shell"
else
    print_warning "Setup completed with warnings! Please check the messages above."
    echo -e "${YELLOW}    Some features might not work as expected.${NC}"
fi
