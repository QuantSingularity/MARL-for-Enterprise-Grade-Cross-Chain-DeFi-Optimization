#!/bin/bash
# CLI Script for MARL Cross-Chain DeFi Demo
# 
# This script runs the complete demo pipeline:
# 1. Generate synthetic data
# 2. Train agents
# 3. Evaluate trained agents
#
# Usage: ./cli.sh [command]
# Commands:
#   all       - Run complete pipeline (default)
#   data      - Generate synthetic data only
#   train     - Train agents only
#   eval      - Evaluate agents only
#   test      - Run unit tests
#   clean     - Clean generated files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Functions
print_header() {
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Check if key packages are installed
    python3 -c "import torch" 2>/dev/null || {
        print_info "Installing dependencies..."
        pip install -q -r requirements.txt
    }
    
    print_success "Dependencies ready"
}

generate_data() {
    print_header "Step 1: Generating Synthetic Data"
    
    python3 src/data/generate_synthetic.py \
        --output-dir data/synthetic \
        --n-steps 1000 \
        --n-swaps 500 \
        --n-chains 2 \
        --n-bridges 2 \
        --seed 42
    
    print_success "Synthetic data generated"
}

train_agents() {
    print_header "Step 2: Training MARL Agents"
    
    print_info "Training QMIX agent..."
    python3 src/train/train_synthetic.py \
        --config configs/demo.yaml \
        --agent qmix \
        --output-dir checkpoints
    
    print_success "QMIX training complete"
    
    print_info "Training MAPPO agent..."
    python3 src/train/train_synthetic.py \
        --config configs/demo.yaml \
        --agent mappo \
        --output-dir checkpoints
    
    print_success "MAPPO training complete"
}

evaluate_agents() {
    print_header "Step 3: Evaluating Trained Agents"
    
    if [ -f "checkpoints/qmix_model.pth" ]; then
        print_info "Evaluating QMIX..."
        python3 src/eval/evaluate_demo.py \
            --checkpoint checkpoints/qmix_model.pth \
            --agent-type qmix \
            --n-episodes 20 \
            --output-dir results \
            --compare-baseline
        print_success "QMIX evaluation complete"
    else
        print_error "QMIX checkpoint not found, skipping evaluation"
    fi
    
    if [ -f "checkpoints/mappo_model.pth" ]; then
        print_info "Evaluating MAPPO..."
        python3 src/eval/evaluate_demo.py \
            --checkpoint checkpoints/mappo_model.pth \
            --agent-type mappo \
            --n-episodes 20 \
            --output-dir results \
            --compare-baseline
        print_success "MAPPO evaluation complete"
    else
        print_error "MAPPO checkpoint not found, skipping evaluation"
    fi
}

run_tests() {
    print_header "Running Unit Tests"
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short
        print_success "Tests passed"
    else
        print_info "pytest not found, running basic import tests..."
        python3 -c "from src.envs.cross_chain_env import CrossChainEnv; print('Environment import OK')"
        python3 -c "from src.agents.qmix import QMIXAgent; print('QMIX import OK')"
        python3 -c "from src.agents.mappo import MAPPOAgent; print('MAPPO import OK')"
        print_success "Basic import tests passed"
    fi
}

run_demo() {
    print_header "Running Environment Demo"
    
    python3 src/envs/demo_env.py --steps 20 --seed 42
    
    print_success "Demo complete"
}

clean_files() {
    print_header "Cleaning Generated Files"
    
    rm -rf data/synthetic
    rm -rf checkpoints/*.pth
    rm -rf results/*.csv results/*.png
    rm -rf __pycache__ src/**/__pycache__
    rm -rf .pytest_cache
    
    print_success "Cleaned generated files"
}

show_help() {
    echo "MARL Cross-Chain DeFi Demo - Command Line Interface"
    echo ""
    echo "Usage: ./cli.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all       Run complete pipeline (default)"
    echo "  data      Generate synthetic data only"
    echo "  train     Train agents only"
    echo "  eval      Evaluate agents only"
    echo "  test      Run unit tests"
    echo "  demo      Run environment demo"
    echo "  clean     Clean generated files"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./cli.sh           # Run complete pipeline"
    echo "  ./cli.sh data      # Generate data only"
    echo "  ./cli.sh train     # Train agents only"
    echo "  ./cli.sh test      # Run tests"
}

# Main execution
main() {
    COMMAND=${1:-all}
    
    echo ""
    echo "  ███╗   ███╗ █████╗ ██████╗ ██╗         ██████╗██████╗  ██████╗ ███████╗███████╗███████╗"
    echo "  ████╗ ████║██╔══██╗██╔══██╗██║        ██╔════╝██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝"
    echo "  ██╔████╔██║███████║██████╔╝██║        ██║     ██████╔╝██║   ██║█████╗  ███████╗███████╗"
    echo "  ██║╚██╔╝██║██╔══██║██╔══██╗██║        ██║     ██╔══██╗██║   ██║██╔══╝  ╚════██║╚════██║"
    echo "  ██║ ╚═╝ ██║██║  ██║██║  ██║███████╗   ╚██████╗██║  ██║╚██████╔╝██║     ███████║███████║"
    echo "  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚══════╝╚══════╝"
    echo ""
    echo "  Cross-Chain Liquidity and Routing Optimization"
    echo ""
    
    case $COMMAND in
        all)
            print_info "Running complete pipeline..."
            check_python
            check_dependencies
            generate_data
            train_agents
            evaluate_agents
            print_header "Pipeline Complete!"
            echo ""
            echo "Results saved to:"
            echo "  - Data: data/synthetic/"
            echo "  - Models: checkpoints/"
            echo "  - Results: results/"
            echo ""
            ;;
        data)
            check_python
            check_dependencies
            generate_data
            ;;
        train)
            check_python
            check_dependencies
            train_agents
            ;;
        eval)
            check_python
            evaluate_agents
            ;;
        test)
            check_python
            run_tests
            ;;
        demo)
            check_python
            run_demo
            ;;
        clean)
            clean_files
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
