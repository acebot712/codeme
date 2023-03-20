#!/usr/bin/env bash

# Set default values
GPUS="all"

# Parse command line arguments
ARGS=$(getopt -o d:o:l:g: --long datasets:,output:,log:,gpus: -n 'docker_run.sh' -- "$@")
eval set -- "$ARGS"

while true; do
    case "$1" in
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
        --)
            shift
            break ;;
        *)
            echo "Internal error!"
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
    -p 6006:6006 \
    --name "${IMAGE}" \
    --cidfile ./${IMAGE}.cid \
    --detach \
    "${IMAGE}"
fi

# ./just-run.sh --datasets /path/to/host/datasets --output /path/to/host/output --log /path/on/host/log --gpus device=0,2
# GPU: sudo docker run -v /opt/data/datasets:/app/datasets -v /opt/data/output:/app/output -v /opt/data/log:/var/log -p 6006:6006 --gpus device="MIG-GPU-3a712174-fefc-7f55-d9fc-1f6d69c3dfb5/5/0" --detach jfrog.fkinternal.com/n200m-common-infra/fine-tune-gpu:0.0.2
