#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default installation directory
DEFAULT_INSTALL_DIR="/opt/icicle/lib/backend/halo2"

# Halo2 repository details
HALO2_REPO="https://github.com/zkonduit/halo2"
HALO2_BRANCH="ac/conditional-compilation-icicle2"

# Parse command line arguments
AUTO_YES=false
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -y, --yes    Automatically answer 'yes' to all prompts"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}EZKL GPU Setup Script${NC}"
echo -e "${GREEN}=====================${NC}"
echo ""

# Parse commit hash from Cargo.lock
echo "Parsing halo2 commit hash from Cargo.lock..."
if [ ! -f "Cargo.lock" ]; then
    echo -e "${RED}Error: Cargo.lock not found. Please run this script from the project root.${NC}"
    exit 1
fi

HALO2_COMMIT=$(grep "github\.com/zkonduit/halo2?" Cargo.lock | grep -v "halo2wrong" | head -1 | grep -o "#[a-f0-9]\{40\}" | cut -c2-)

if [ -z "$HALO2_COMMIT" ]; then
    echo -e "${RED}Error: Could not parse halo2 commit hash from Cargo.lock${NC}"
    exit 1
fi

echo -e "${GREEN}Found halo2 commit: $HALO2_COMMIT${NC}"
echo ""
echo "This script will:"
echo "1. Sparse checkout the halo2 repository at commit $HALO2_COMMIT"
echo "2. Extract only the icicle/backend/cuda/ directory"
echo "3. Set the ICICLE_BACKEND_INSTALL_DIR environment variable"
echo ""

# Check if user wants to override the default directory
if [ "$AUTO_YES" = true ]; then
    INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    echo -e "${GREEN}Using default installation directory: ${INSTALL_DIR}${NC}"
else
    echo -e "${YELLOW}Default installation directory: ${DEFAULT_INSTALL_DIR}${NC}"
    read -p "Do you want to use a different directory? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter the installation directory: " INSTALL_DIR
        INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"  # Expand ~ to $HOME
    else
        INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    fi

    # Confirm the installation directory
    echo ""
    echo -e "${YELLOW}Installation directory: ${INSTALL_DIR}${NC}"
    read -p "Continue with this directory? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Setup cancelled by user.${NC}"
        exit 1
    fi
fi

# Check if ICICLE_BACKEND_INSTALL_DIR is already set
if [ ! -z "$ICICLE_BACKEND_INSTALL_DIR" ] && [ "$AUTO_YES" = false ]; then
    echo ""
    echo -e "${YELLOW}Warning: ICICLE_BACKEND_INSTALL_DIR is already set to: $ICICLE_BACKEND_INSTALL_DIR${NC}"
    read -p "Do you want to override it? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Setup cancelled by user.${NC}"
        exit 1
    fi
elif [ ! -z "$ICICLE_BACKEND_INSTALL_DIR" ] && [ "$AUTO_YES" = true ]; then
    echo -e "${GREEN}Overriding existing ICICLE_BACKEND_INSTALL_DIR (was: $ICICLE_BACKEND_INSTALL_DIR)${NC}"
fi

echo ""
echo -e "${GREEN}Starting GPU setup...${NC}"

# Create installation directory
echo "Creating installation directory..."
mkdir -p "$INSTALL_DIR"

# Create temporary directory for sparse checkout
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Clone with sparse checkout
echo "Cloning halo2 repository with sparse checkout..."
cd "$TEMP_DIR"
git clone --filter=blob:none --sparse "$HALO2_REPO" halo2
cd halo2

# Checkout the specific branch and commit
echo "Checking out branch $HALO2_BRANCH at commit $HALO2_COMMIT..."
git checkout "$HALO2_BRANCH"
git checkout "$HALO2_COMMIT"

# Configure sparse checkout
echo "Configuring sparse checkout for icicle/backend/cuda/..."
git sparse-checkout init --cone
git sparse-checkout set icicle/backend/cuda/

# Copy the icicle directory to the installation location
if [ -d "icicle/backend/cuda" ]; then
    echo "Copying icicle/backend/cuda/ to $INSTALL_DIR..."
    cp -r icicle/backend/cuda/* "$INSTALL_DIR/"
    echo -e "${GREEN}Files copied successfully!${NC}"
else
    echo -e "${RED}Error: icicle/backend/cuda directory not found in the repository${NC}"
    exit 1
fi

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Ask user about setting environment variable permanently
SETUP_PERMANENT_ENV=false
if [ "$AUTO_YES" = true ]; then
    SETUP_PERMANENT_ENV=true
    echo ""
    echo -e "${GREEN}Setting ICICLE_BACKEND_INSTALL_DIR environment variable permanently...${NC}"
else
    echo ""
    echo -e "${YELLOW}Do you want to set ICICLE_BACKEND_INSTALL_DIR environment variable permanently?${NC}"
    echo "This will add 'export ICICLE_BACKEND_INSTALL_DIR=\"$INSTALL_DIR\"' to your shell configuration file."
    read -p "Set environment variable permanently? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SETUP_PERMANENT_ENV=true
    fi
fi

if [ "$SETUP_PERMANENT_ENV" = true ]; then
    echo "Setting ICICLE_BACKEND_INSTALL_DIR environment variable..."

    # Detect shell and set environment variable accordingly
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    else
        # Try to detect based on $SHELL
        case "$SHELL" in
            */zsh)
                SHELL_RC="$HOME/.zshrc"
                ;;
            */bash)
                SHELL_RC="$HOME/.bashrc"
                ;;
            *)
                SHELL_RC="$HOME/.profile"
                ;;
        esac
    fi

    # Add environment variable to shell configuration
    ENV_EXPORT="export ICICLE_BACKEND_INSTALL_DIR=\"$INSTALL_DIR\""

    # Check if the variable is already set in the file
    if [ -f "$SHELL_RC" ] && grep -q "ICICLE_BACKEND_INSTALL_DIR" "$SHELL_RC"; then
        # Replace existing line
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|export ICICLE_BACKEND_INSTALL_DIR=.*|$ENV_EXPORT|" "$SHELL_RC"
        else
            # Linux
            sed -i "s|export ICICLE_BACKEND_INSTALL_DIR=.*|$ENV_EXPORT|" "$SHELL_RC"
        fi
        echo "Updated existing ICICLE_BACKEND_INSTALL_DIR in $SHELL_RC"
    else
        # Add new line
        echo "$ENV_EXPORT" >> "$SHELL_RC"
        echo "Added ICICLE_BACKEND_INSTALL_DIR to $SHELL_RC"
    fi

    echo -e "${GREEN}Environment variable set permanently.${NC}"
else
    echo "Skipping permanent environment variable setup."
fi

# Export for current session regardless
export ICICLE_BACKEND_INSTALL_DIR="$INSTALL_DIR"
echo "Environment variable set for current session."

echo ""
echo -e "${GREEN}GPU setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "1. The ICICLE_BACKEND_INSTALL_DIR environment variable has been set to: $INSTALL_DIR"
if [ "$SETUP_PERMANENT_ENV" = true ]; then
    echo "2. Please restart your terminal or run: source $SHELL_RC"
else
    echo "2. To use GPU features, set: export ICICLE_BACKEND_INSTALL_DIR=\"$INSTALL_DIR\""
fi
echo "3. You can now build with GPU support using: cargo build --features gpu-accelerated"
echo ""
echo -e "${GREEN}Setup complete!${NC}"