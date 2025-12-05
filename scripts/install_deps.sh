#!/bin/bash
# RM Abstract Layer - Dependency Installation Script
# 
# Usage:
#   ./scripts/install_deps.sh [component]
#
# Components:
#   all         - Install all components
#   base        - Base requirements only
#   gpu         - GPU/vLLM support
#   triton      - Triton Inference Server
#   torchserve  - TorchServe
#   npu-rbln    - Rebellions ATOM NPU
#   java        - Java (required for TorchServe server)
#   docker      - Docker (required for Triton server)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_DIR="$PROJECT_DIR/requirements"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Detect package manager
detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        echo "apt"
    elif command -v yum &> /dev/null; then
        echo "yum"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v brew &> /dev/null; then
        echo "brew"
    else
        echo "unknown"
    fi
}

# Install Java
install_java() {
    print_header "Installing Java (OpenJDK 11)"
    
    PKG_MANAGER=$(detect_package_manager)
    
    case $PKG_MANAGER in
        apt)
            sudo apt-get update
            sudo apt-get install -y openjdk-11-jdk
            ;;
        yum|dnf)
            sudo $PKG_MANAGER install -y java-11-openjdk java-11-openjdk-devel
            ;;
        brew)
            brew install openjdk@11
            ;;
        *)
            print_error "Unknown package manager. Please install Java manually."
            return 1
            ;;
    esac
    
    # Verify installation
    if java -version 2>&1 | grep -q "11"; then
        print_success "Java 11 installed successfully"
    else
        print_warning "Java installed but version may not be 11"
    fi
}

# Install Docker
install_docker() {
    print_header "Installing Docker"
    
    if command -v docker &> /dev/null; then
        print_success "Docker already installed"
        docker --version
        return 0
    fi
    
    PKG_MANAGER=$(detect_package_manager)
    
    case $PKG_MANAGER in
        apt)
            # Install Docker using official script
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            rm get-docker.sh
            
            # Add current user to docker group
            sudo usermod -aG docker $USER
            print_warning "Please log out and back in for docker group changes to take effect"
            ;;
        yum|dnf)
            sudo $PKG_MANAGER install -y docker
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        brew)
            brew install --cask docker
            print_warning "Please start Docker Desktop manually"
            ;;
        *)
            print_error "Unknown package manager. Please install Docker manually."
            print_warning "Visit: https://docs.docker.com/get-docker/"
            return 1
            ;;
    esac
    
    print_success "Docker installed"
}

# Install NVIDIA Container Toolkit (for GPU support in Docker)
install_nvidia_docker() {
    print_header "Installing NVIDIA Container Toolkit"
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "NVIDIA driver not detected. Skipping NVIDIA Container Toolkit."
        return 0
    fi
    
    PKG_MANAGER=$(detect_package_manager)
    
    if [ "$PKG_MANAGER" = "apt" ]; then
        # Add NVIDIA repository
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        
        print_success "NVIDIA Container Toolkit installed"
    else
        print_warning "Please install NVIDIA Container Toolkit manually for your distribution"
        print_warning "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

# Install Python requirements
install_python_requirements() {
    local component=$1
    print_header "Installing Python requirements: $component"
    
    local req_file="$REQUIREMENTS_DIR/$component.txt"
    
    if [ ! -f "$req_file" ]; then
        print_error "Requirements file not found: $req_file"
        return 1
    fi
    
    # Use pip or uv
    if command -v uv &> /dev/null; then
        uv pip install -r "$req_file"
    else
        pip install -r "$req_file"
    fi
    
    print_success "Python requirements installed: $component"
}

# Pull Triton Docker image
setup_triton_docker() {
    print_header "Setting up Triton Inference Server Docker"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed. Run: $0 docker"
        return 1
    fi
    
    echo "Pulling Triton Inference Server image..."
    docker pull nvcr.io/nvidia/tritonserver:24.01-py3
    
    print_success "Triton Docker image pulled"
    echo ""
    echo "To start Triton server:"
    echo "  docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \\"
    echo "    -v /path/to/models:/models \\"
    echo "    nvcr.io/nvidia/tritonserver:24.01-py3 \\"
    echo "    tritonserver --model-repository=/models"
}

# Show usage
show_usage() {
    echo "RM Abstract Layer - Dependency Installation Script"
    echo ""
    echo "Usage: $0 [component]"
    echo ""
    echo "Components:"
    echo "  all           Install all components (recommended)"
    echo "  base          Base requirements only"
    echo "  gpu           GPU/vLLM support"
    echo "  triton        Triton client + Docker setup"
    echo "  torchserve    TorchServe + Java"
    echo "  npu-rbln      Rebellions ATOM NPU (manual steps)"
    echo ""
    echo "System dependencies:"
    echo "  java          Install Java 11 (for TorchServe)"
    echo "  docker        Install Docker (for Triton server)"
    echo "  nvidia-docker Install NVIDIA Container Toolkit"
    echo ""
    echo "Examples:"
    echo "  $0 all                    # Install everything"
    echo "  $0 gpu                    # Install GPU/vLLM only"
    echo "  $0 triton docker          # Install Triton + Docker"
    echo "  $0 torchserve java        # Install TorchServe + Java"
}

# Main
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    print_header "RM Abstract Layer - Dependency Installer"
    
    for component in "$@"; do
        case $component in
            all)
                install_python_requirements "all"
                install_java
                install_docker
                install_nvidia_docker
                setup_triton_docker
                ;;
            base)
                install_python_requirements "base"
                ;;
            gpu)
                install_python_requirements "gpu"
                ;;
            triton)
                install_python_requirements "triton"
                ;;
            torchserve)
                install_python_requirements "torchserve"
                ;;
            npu-rbln)
                install_python_requirements "npu-rbln"
                echo ""
                print_warning "For RBLN NPU, you also need:"
                echo "  1. Rebellions ATOM NPU hardware"
                echo "  2. RBLN SDK from https://docs.rbln.ai/"
                echo "  3. pip install vllm-rbln  OR  pip install optimum-rbln"
                ;;
            java)
                install_java
                ;;
            docker)
                install_docker
                ;;
            nvidia-docker)
                install_nvidia_docker
                ;;
            triton-docker)
                setup_triton_docker
                ;;
            help|--help|-h)
                show_usage
                ;;
            *)
                print_error "Unknown component: $component"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo ""
    print_success "Installation complete!"
    echo ""
    echo "Verify installation:"
    echo "  python -m rm_abstract.system_validator"
}

main "$@"

