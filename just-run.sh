#!/usr/bin/env bash

# Set default values
GPUS="all"
TRAIN_CSV="datasets/codesearchnet_train_py_small.csv"
EVAL_CSV="datasets/codesearchnet_valid_py_small.csv"
DEVICE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -d|--datasets)
            DATASETS_DIR="$2" # Directory containing datasets
            shift 2 ;;
        -o|--output)
            OUTPUT_DIR="$2" # Directory for output files
            shift 2 ;;
        -l|--log)
            LOG_DIR="$2" # Directory for log files
            shift 2 ;;
        -g|--gpus)
            GPUS="$2" # GPUs to use
            shift 2 ;;
        -t|--train-csv)
            TRAIN_CSV="$2" # Path to training CSV file
            shift 2 ;;
        -e|--eval-csv)
            EVAL_CSV="$2" # Path to evaluation CSV file
            shift 2 ;;
        -v|--device)
            DEVICE="$2" # Device to use (cpu or cuda)
            shift 2 ;;
        *)
            echo "Unknown option: $key"
            exit 1 ;;
    esac
done

# Run docker container
IMAGE=$(yq r image.yaml image)

if command -v nvidia-smi &> /dev/null
then
    docker run \
    --platform=linux/amd64 \
    --gpus "${GPUS}" \
    -v "${DATASETS_DIR}":/app/datasets \
    -v "${OUTPUT_DIR}":/app/output \
    -v "${LOG_DIR}":/app/log \
    -e TRAIN_CSV="${TRAIN_CSV}" \
    -e EVAL_CSV="${EVAL_CSV}" \
    -e DEVICE="${DEVICE}" \
    -p 6006:6006 \
    --name "${IMAGE}" \
    --cidfile ./${IMAGE}.cid \
    --detach \
    "${IMAGE}"
else
    docker run \
    --platform=linux/amd64 \
    -v "${DATASETS_DIR}":/app/datasets \
    -v "${OUTPUT_DIR}":/app/output \
    -v "${LOG_DIR}":/app/log \
    -e TRAIN_CSV="${TRAIN_CSV}" \
    -e EVAL_CSV="${EVAL_CSV}" \
    -e DEVICE="${DEVICE}" \
    -p 6006:6006 \
    --name "${IMAGE}" \
    --cidfile ./${IMAGE}.cid \
    --detach \
    "${IMAGE}"
fi


# ./just-run.sh --datasets /path/to/host/datasets --output /path/to/host/output --log /path/on/host/log --gpus device=0,2
# docker run \
#     -v /opt/data/datasets:/app/datasets \
#     -v /opt/data/output:/app/output \
#     -v /opt/data/log:/var/log \
#     -v /opt/data/models:/app/models \
#     -e TRAIN_CSV=datasets/codesearchnet_train_py_small.csv \
#     -e EVAL_CSV=datasets/codesearchnet_valid_py_small.csv \
#     -e DEVICE=cuda \
#     -p 6006:6006 \
#     --gpus device="MIG-GPU-3a712174-fefc-7f55-d9fc-1f6d69c3dfb5/5/0" \
#     --detach \
#     jfrog.fkinternal.com/n200m-common-infra/fine-tune-gpu:0.0.6
