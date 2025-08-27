export AWS_DEFAULT_REGION=us-east-1
REGION=us-east-1
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin 684288478426.dkr.ecr.${REGION}.amazonaws.com

RANDOM_TAG=$(printf '%s' $(echo "$RANDOM" | md5sum) | cut -c 1-24)
DOCKER_IMAGE_TAG=nemo-eval
WORKSPACE=/workspace/ACEBench/

sudo docker build \
    --build-arg WORKSPACE=${WORKSPACE} \
    --tag=$DOCKER_IMAGE_TAG:${RANDOM_TAG} \
    -f Dockerfile_acebench.local .
echo "Built docker image nemo-eval:${RANDOM_TAG}"

# Change -v /fsx-pretraining:/fsx-pretraining \ if you have different mnt.
echo "Running docker image nemo-eval:${RANDOM_TAG}"
sudo docker run --gpus all -p 8661:8661 \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --name=$DOCKER_IMAGE_TAG-${RANDOM_TAG} \
    -v /fsx-pretraining:/fsx-pretraining \
    -w ${WORKSPACE} \
    --mount type=tmpfs,destination=/tmpfs $DOCKER_IMAGE_TAG:${RANDOM_TAG} \
    bash
