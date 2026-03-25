#!/bin/bash

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -r, --rebuild    Clean and rebuild from scratch"
    echo "  -h, --help       Show this help message"
    exit 1
}

REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rebuild)
            REBUILD=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ "$REBUILD" = true ]; then
    echo "Cleaning TransformerEngine..."
    pip uninstall transformer_engine -y
    rm -rf build
    rm -rf transformer_engine.egg-info
    rm -f transformer_engine/transformer_engine_torch.cpython-310-x86_64-linux-gnu.so
    rm -f libtransformer_engine.so
    rm -f transformer_engine_torch.cpython-310-x86_64-linux-gnu.so
fi

echo "Installing TransformerEngine..."
export NVTE_FRAMEWORK=musa
pip install --no-build-isolation -v . 2>&1 | tee install.log
