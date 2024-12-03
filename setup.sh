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

# Check if git-lfs is installed
print_section "Git LFS Setup"
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs not found. Installing..."
    
    # Check the operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install git-lfs
        else
            echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
            sudo apt-get install git-lfs
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
            sudo yum install git-lfs
        else
            echo "Could not detect package manager. Please install git-lfs manually."
            exit 1
        fi
    else
        echo "Unsupported operating system. Please install git-lfs manually."
        exit 1
    fi
    
    # Initialize git-lfs
    git-lfs install
    print_success "git-lfs installed and initialized"
else
    print_success "git-lfs already installed"
fi

# Initialize and update git submodules
print_section "Git Submodules"
echo "Initializing git submodules..."
git submodule update --init --recursive
print_success "Git submodules initialized"

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
    poetry install --with dev --no-root
    print_success "Poetry environment created successfully"
else
    print_success "Poetry environment already exists"
fi

# ---- PRE-COMMIT SETUP ---- #
print_section "Pre-commit Setup"

# First check if pre-commit is installed in the poetry environment
if ! poetry run pre-commit --version &> /dev/null; then
    echo "Installing pre-commit in poetry environment..."
    poetry add pre-commit --group dev
    print_success "pre-commit installed successfully"
else
    print_success "pre-commit already installed"
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
poetry run pre-commit install
print_success "Pre-commit hooks installed"

# Run pre-commit hooks on all files
echo "Running pre-commit hooks on all files..."
poetry run pre-commit run --all-files
print_success "Pre-commit initial run complete"

# ---- EVALUATION SETUP ---- #
# Clone Paloma dataset if credentials are provided and directory doesn't exist
print_section "Evaluation Setup"
if [ ! -d "lib/paloma" ]; then
    if [ ! -z "$HF_TOKEN" ]; then
        echo "Cloning Paloma evaluation dataset..."
        git clone https://oauth2:${HF_TOKEN}@huggingface.co/datasets/allenai/paloma lib/paloma
        print_success "Paloma dataset cloned successfully"
    else
        print_warning "Skipping Paloma dataset clone. To clone, provide valid HuggingFace credentials"
        echo -e "${YELLOW}Note: You need to request access to the Paloma dataset on HuggingFace:${NC}"
        echo -e "${BLUE}https://huggingface.co/datasets/allenai/paloma${NC}"
        echo -e "${YELLOW}Visit the dataset page and click 'Access Request' to request permission.${NC}"
    fi
else
    print_success "Paloma dataset already exists, skipping clone"
fi

# Create environment for running evaluation inside of lib/olmo_eval
# skip if already exists
if [ ! -d "lib/olmo-eval/env" ]; then
    print_section "OLMo Eval Setup"
    poetry shell
    cd lib/olmo-eval
    echo "Creating virtual environment..."
    virtualenv env
    source env/bin/activate
    pip install -e .
    deactivate
    cd ../../
    print_success "OLMo eval environment setup complete"
    exit # out of poetry shell
else
    print_success "OLMo eval environment already exists, skipping setup"
fi

print_section "Setup Complete! 🎉"
echo "You can now activate the poetry environment with: poetry shell"

