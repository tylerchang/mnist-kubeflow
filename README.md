# MNIST Project Kubeflow Deployment Guide

This project implements a deep learning workflow for MNIST digit recognition using Kubeflow on Google’s Kubernetes Engine (GKE).

## Prerequisites
- Google Cloud SDK
- Docker
- Kubernetes access

## Setup and Deployment Steps

### Create GKE Cluster
The following will create a cluster called `kubeflow-cluster`. May need to change region/zone based on availability.
```bash
gcloud container clusters create kubeflow-cluster --machine-type n1-standard-4 --num-nodes 2 --zone us-central1-a
```

Get credentials to interact with the cluster
```bash
gcloud container clusters get-credentials kubeflow-cluster --zone us-central1-a
```

### Install Kubernetes Controller
```bash
brew install kubectl
```

### Install Kubeflow Training Operator
```bash
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```

- Verify Kubeflow installation with `kubectl get pods -n kubeflow`

### Set Project Configuration
```bash
# Navigate to project directory
# Set your Google Cloud project ID
export PROJECT_ID=[GCP PROJECT ID]
export GCR_PATH=gcr.io/$PROJECT_ID
```

Also navigate to `inference-deployment.yaml` and `train-tf-job.yaml`, and look for:
```bash
image: gcr.io/<PROJECT_ID>/mnist-inference:latest
```
Change `<PROJECT_ID>` in both files to your own from Google Cloud.

### Docker Authentication
```bash
gcloud auth configure-docker
```

### PVC Setup
Creates a PVC called `mnist-model-pvc` through the yaml file
```bash
kubectl apply -f pvc.yaml
```

### Build and Push Training Image
```bash
docker build --platform linux/amd64 -t $GCR_PATH/mnist-training:latest -f Dockerfile.train .
```
```bash
docker push $GCR_PATH/mnist-training:latest
```

### Build and Push Inference Image
```bash
docker build --platform linux/amd64 -t $GCR_PATH/mnist-inference:latest -f Dockerfile.inference .
```
```bash
docker push $GCR_PATH/mnist-inference:latest
```

### Deploy Training Job
```bash
kubectl apply -f train-tf-job.yaml
```

Check progression and status with:
```bash
kubectl get pods | grep mnist-training
kubectl logs job/mnist-training-job
```

### Deploy Inference Service
```bash
kubectl apply -f inference-deployment.yaml
```

```bash
kubectl apply -f inference-service.yaml
```

### Get Service URL (Look for EXTERNAL IP)
```bash
kubectl get service mnist-inference-service
```

Enter the external URL into the web browser to access the application. The `sample digits` directory provides some testing images for use.

# Screen Captures
![Alt text for image](screenshots/1.png)
![Alt text for image](screenshots/2.png)
![Alt text for image](screenshots/3.png)

# Shutting Down Safely

### Delete Inference Resources
```bash
# Delete the inference deployment
kubectl delete deployment mnist-inference

# Delete the inference service
kubectl delete service mnist-inference-service
```

### Delete Training Job
```bash
# Remove the TFJob for training
kubectl delete tfjob mnist-tfjob
```

### Clean Up Persistent Volume Claim (Optional)
```bash
# Remove the persistent volume claim if you want to fully clean up
kubectl delete pvc mnist-model-pvc
```

### Verify Resource Deletion
```bash
# Check that no pods are running
kubectl get pods

# Confirm no deployments remain
kubectl get deployments

# Verify services are removed
kubectl get services
```