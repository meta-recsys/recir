# TPU Cluster Management Runbook (GKE)

> **Last updated:** 2026-04-17
> **Owner:** linjianma@meta.com
> **Project:** MRS Core Modeling - Model Co-design

This runbook covers the full lifecycle of managing TPU v7x clusters on Google Kubernetes
Engine (GKE) within the `mrs-tpu-y317h` GCP project. It is intended as a self-contained
reference for engineers and AI coding assistants (e.g., Claude Code sessions).

---

## Architecture Overview

```
Local Mac (linjianma)
  │
  ├── SSH via IAP ──► linjian-client VM (GCE, us-central1-c)
  │                     ├── gcloud CLI (service account: 620069540265-compute@developer.gserviceaccount.com)
  │                     ├── kubectl (configured via gcloud)
  │                     ├── docker (for building images)
  │                     └── ~/RecML/ (training code repo)
  │
  └── kubectl (local) ──► GKE clusters (for RBAC-sensitive operations)
                            ├── recir-v7-8-1 (1 TPU node, 4 chips)
                            └── recir-v7-8-2 (2 TPU nodes, 8 chips)
```

### Key details

| Parameter | Value |
|---|---|
| **GCP Project** | `mrs-tpu-y317h` |
| **Region** | `us-central1` |
| **TPU Zone** | `us-central1-ai1a` |
| **VPC Network** | `mrs-tpu-testvpc-privatenetwork` |
| **Subnet** | `mrs-tpu-testvpc-privatesubnet` |
| **TPU Reservation** | `batch2pod2s` |
| **Reservation capacity** | 9,216 TPU v7x chips (expires 2031-04-08) |
| **TPU machine type** | `tpu7x-standard-4t` (4 chips per node) |
| **Container image** | `us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest` |
| **Container registry** | `us-central1-docker.pkg.dev/mrs-tpu-y317h/recml` |
| **Client VM** | `linjian-client` (GCE instance, `us-central1-c`) |

### SSH access to linjian-client

From local Mac:
```bash
gcloud compute ssh --project=mrs-tpu-y317h --zone=us-central1-c linjian-client --tunnel-through-iap
```

Or simply (if `~/.ssh/config` is configured):
```bash
ssh linjian-client
```

The SSH config entry (in `~/.ssh/config`):
```
Host linjian-client
    ProxyCommand /Users/linjianma/Downloads/google-cloud-sdk/bin/gcloud compute ssh linjian-client --project=mrs-tpu-y317h --zone=us-central1-c --tunnel-through-iap --ssh-flag="-W %h:%p"
    User linjianma
    IdentityFile ~/.ssh/google_compute_engine
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

### Environment variables (set these first)

On `linjian-client` or in any script:
```bash
export PROJECT_ID=mrs-tpu-y317h
export REGION=us-central1
export ZONE=us-central1-ai1a
export REPO=recml
export IMAGE_PATH=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/recml_base_image:latest
```

---

## 1. Check Cluster Information

All commands run from `linjian-client` unless noted otherwise.

### List all clusters

```bash
gcloud container clusters list --project=$PROJECT_ID
```

### Describe a specific cluster

```bash
gcloud container clusters describe <CLUSTER_NAME> \
  --region=$REGION \
  --project=$PROJECT_ID
```

### Get kubectl credentials

```bash
gcloud container clusters get-credentials <CLUSTER_NAME> \
  --region=$REGION \
  --project=$PROJECT_ID
```

### Switch kubectl context between clusters

The context name follows the pattern `gke_<PROJECT>_<REGION>_<CLUSTER>`:

```bash
# Switch to recir-v7-8-1
kubectl config use-context gke_mrs-tpu-y317h_us-central1_recir-v7-8-1

# Switch to recir-v7-8-2
kubectl config use-context gke_mrs-tpu-y317h_us-central1_recir-v7-8-2

# Check current context
kubectl config current-context
```

### Check nodes

```bash
# All nodes with details
kubectl get nodes -o wide

# Only TPU nodes
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x -o wide
```

### List node pools

```bash
gcloud container node-pools list \
  --cluster=<CLUSTER_NAME> \
  --region=$REGION \
  --project=$PROJECT_ID
```

### Check running pods

```bash
# All namespaces
kubectl get pods -A

# Default namespace with wide output
kubectl get pods -o wide

# Watch status in real time
kubectl get pods -o wide -w
```

### Check TPU reservation capacity

```bash
gcloud compute reservations describe batch2pod2s \
  --project=$PROJECT_ID \
  --zone=$ZONE
```

Key fields in output:
- `reservedResources.accelerator.acceleratorCount` — total chips available (9,216)
- `inUseResources.accelerator.acceleratorCount` — chips currently in use

### Check TPU topology on nodes

```bash
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x \
  -o jsonpath='{range .items[*]}{.metadata.name}: topology={.metadata.labels.cloud\.google\.com/gke-tpu-topology}{"\n"}{end}'
```

---

## 2. Create and Delete Clusters

### Create a new cluster with TPU nodes

This is a 3-step process: create the GKE cluster, add the TPU node pool, then install
required CRDs. Total time: ~10 minutes.

#### Step 1: Create the GKE cluster (CPU default pool)

Run on `linjian-client`:

```bash
gcloud container clusters create <CLUSTER_NAME> \
  --project=$PROJECT_ID \
  --region=$REGION \
  --node-locations=$ZONE \
  --network=mrs-tpu-testvpc-privatenetwork \
  --subnetwork=mrs-tpu-testvpc-privatesubnet \
  --release-channel=rapid \
  --num-nodes=6 \
  --machine-type=e2-standard-16 \
  --scopes=storage-full \
  --enable-ip-alias \
  --autoscaling-profile=optimize-utilization \
  --enable-autoscaling --total-min-nodes=1 --total-max-nodes=1000
```

Takes ~5 minutes. This creates a regional cluster with 6 CPU nodes (`e2-standard-16`)
that handle system workloads (kube-system, scheduling, networking). The autoscaler with
`optimize-utilization` will scale down unused CPU nodes.

**Flag breakdown:**
- `--node-locations=$ZONE` — places nodes in `us-central1-ai1a` (where TPU reservation exists)
- `--network/--subnetwork` — must use the project's VPC for TPU connectivity
- `--release-channel=rapid` — required for latest TPU support
- `--scopes=storage-full` — allows reading/writing to GCS buckets
- `--enable-ip-alias` — required for VPC-native networking

#### Step 2: Add the TPU node pool

```bash
gcloud container node-pools create <CLUSTER_NAME>-np-0 \
  --project=$PROJECT_ID \
  --region=$REGION \
  --cluster=<CLUSTER_NAME> \
  --node-locations=$ZONE \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=<NUM_TPU_NODES> \
  --scopes=cloud-platform,storage-full \
  --reservation-affinity=specific \
  --reservation=batch2pod2s \
  --enable-gvnic
```

Takes ~4 minutes. Each `tpu7x-standard-4t` node has **4 TPU v7x chips**.

**Flag breakdown:**
- `--machine-type=tpu7x-standard-4t` — 4 TPU v7x chips per node
- `--reservation-affinity=specific --reservation=batch2pod2s` — uses the reserved capacity
- `--scopes=cloud-platform,storage-full` — broad access needed for TPU runtime
- `--enable-gvnic` — Google Virtual NIC, improves network performance for TPU traffic

**Sizing reference:**

| `--num-nodes` | Total chips | Topology | Use case |
|---|---|---|---|
| 1 | 4 | `2x2x1` | Small experiments, debugging |
| 2 | 8 | `2x2x1` | Multi-host training |
| 4 | 16 | `4x4x4` | Larger models |
| 16 | 64 | `4x4x4` | Full DLRM training |

#### Step 3: Install JobSet CRD

**⚠️ Must be run from your LOCAL Mac** (the VM service account lacks RBAC permissions):

```bash
# Get credentials locally
gcloud container clusters get-credentials <CLUSTER_NAME> \
  --region=us-central1 \
  --project=mrs-tpu-y317h

# Install JobSet CRD
kubectl apply --server-side \
  -f https://github.com/kubernetes-sigs/jobset/releases/latest/download/manifests.yaml \
  --context=gke_mrs-tpu-y317h_us-central1_<CLUSTER_NAME>
```

JobSet is the Kubernetes API for co-scheduling multi-pod workloads on TPU slices.
Without it, you cannot run multi-node TPU jobs.

#### Step 4: Verify

```bash
# Check all nodes are Ready
kubectl get nodes -o wide

# Check TPU topology
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x \
  -o jsonpath='{range .items[*]}{.metadata.name}: {.metadata.labels.cloud\.google\.com/gke-tpu-topology}{"\n"}{end}'

# Check JobSet CRD is installed
kubectl get crd jobsets.jobset.x-k8s.io
```

You should see:
- CPU nodes (`gke-<CLUSTER_NAME>-default-pool-*`) — some may show `NotReady,SchedulingDisabled` (autoscaler scaling down, this is normal)
- TPU nodes (`gke-tpu-*`) — all should be `Ready`

#### Complete example: Create `recir-v7-8-2` with 2 TPU nodes

```bash
# Step 1: Create cluster (~5 min)
gcloud container clusters create recir-v7-8-2 \
  --project=mrs-tpu-y317h \
  --region=us-central1 \
  --node-locations=us-central1-ai1a \
  --network=mrs-tpu-testvpc-privatenetwork \
  --subnetwork=mrs-tpu-testvpc-privatesubnet \
  --release-channel=rapid \
  --num-nodes=6 \
  --machine-type=e2-standard-16 \
  --scopes=storage-full \
  --enable-ip-alias \
  --autoscaling-profile=optimize-utilization \
  --enable-autoscaling --total-min-nodes=1 --total-max-nodes=1000

# Step 2: Add 2 TPU nodes, 8 chips (~4 min)
gcloud container node-pools create recir-v7-8-2-np-0 \
  --project=mrs-tpu-y317h \
  --region=us-central1 \
  --cluster=recir-v7-8-2 \
  --node-locations=us-central1-ai1a \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=2 \
  --scopes=cloud-platform,storage-full \
  --reservation-affinity=specific \
  --reservation=batch2pod2s \
  --enable-gvnic

# Step 3: Install JobSet (run from LOCAL Mac)
gcloud container clusters get-credentials recir-v7-8-2 \
  --region=us-central1 --project=mrs-tpu-y317h
kubectl apply --server-side \
  -f https://github.com/kubernetes-sigs/jobset/releases/latest/download/manifests.yaml \
  --context=gke_mrs-tpu-y317h_us-central1_recir-v7-8-2

# Step 4: Verify
kubectl get nodes -o wide
```

### Delete a cluster

```bash
gcloud container clusters delete <CLUSTER_NAME> \
  --region=$REGION \
  --project=$PROJECT_ID
```

This is destructive and deletes all node pools, workloads, and data on the cluster.
The TPU reservation capacity is returned to the pool after deletion.

### Delete just a TPU node pool (keep the cluster)

```bash
gcloud container node-pools delete <CLUSTER_NAME>-np-0 \
  --cluster=<CLUSTER_NAME> \
  --region=$REGION \
  --project=$PROJECT_ID
```

---

## 3. Deploy Workloads to the Cluster

### Overview: How TPU workloads run on GKE

TPU workloads use the **JobSet** API to co-schedule pods across TPU nodes. Each pod gets
a full node (4 chips). All chips across nodes in a TPU slice are connected via ICI
(Inter-Chip Interconnect), so JAX sees them as a single unified device mesh.

```
JobSet
  └── Job (worker)
        ├── Pod 0 → TPU Node 1 (4 chips)
        └── Pod 1 → TPU Node 2 (4 chips)
                     ↕ ICI interconnect ↕
              JAX sees all 8 chips as one device mesh
```

### Required YAML fields for TPU workloads

Every TPU workload YAML **must** include:

| Field | Value | Why |
|---|---|---|
| `resources.limits.google.com/tpu` | `4` | Requests all 4 chips on each node |
| `nodeSelector.cloud.google.com/gke-tpu-accelerator` | `tpu7x` | Schedules on TPU nodes |
| `nodeSelector.cloud.google.com/gke-tpu-topology` | e.g. `2x2x1` | **Required by GKE** — must match node topology |
| `ports.containerPort` | `8471` | TPU inter-host communication port |
| `spec.parallelism` / `spec.completions` | `<NUM_TPU_NODES>` | Number of pods = number of TPU nodes |
| `restartPolicy` | `Never` | TPU jobs should not auto-restart |

### Build and push a container image

On `linjian-client`:

```bash
cd ~/RecML

# Build
docker build -t recml-training .

# Tag
docker tag recml-training:latest $IMAGE_PATH

# Authenticate (first time only)
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push
docker push $IMAGE_PATH
```

The Dockerfile is at `~/RecML/Dockerfile`. It uses Python 3.12 with JAX, protobuf,
and `jax-tpu-embedding` pre-installed.

### Deploy a multi-node TPU workload (JobSet)

Example for a 2-node (8-chip) workload:

```yaml
# tpu-workload.yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: <WORKLOAD_NAME>
  namespace: default
spec:
  replicatedJobs:
  - name: worker
    replicas: 1
    template:
      spec:
        parallelism: 2       # ← number of TPU nodes
        completions: 2        # ← must match parallelism
        template:
          spec:
            containers:
            - name: training
              image: us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest
              command: ["python", "<your_script.py>"]
              ports:
              - containerPort: 8471
              resources:
                limits:
                  google.com/tpu: 4
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 2x2x1   # ← check with topology command
            restartPolicy: Never
```

### Submit a workload

From your **local Mac** (recommended, avoids file-copy step):

```bash
# Submit to recir-v7-8-2
kubectl apply -f /Users/linjianma/Documents/tpu/tpu-test-job.yaml \
  --context=gke_mrs-tpu-y317h_us-central1_recir-v7-8-2
```

Or from **linjian-client** (copy file first):

```bash
# Copy YAML to VM
scp /Users/linjianma/Documents/tpu/tpu-test-job.yaml linjian-client:~/tpu-test-job.yaml

# Submit on VM
ssh linjian-client "kubectl apply -f ~/tpu-test-job.yaml"
```

### Monitor workload status

```bash
# Watch pods in real time (Ctrl+C to stop)
kubectl get pods -o wide -w

# Check JobSet status
kubectl get jobsets

# Describe for scheduling events/errors
kubectl describe jobset <WORKLOAD_NAME>
```

### Check outputs / logs

```bash
# List pods to get their full names
kubectl get pods

# View logs from a completed or running pod
kubectl logs <pod-name>

# Stream logs in real time (while pod is running)
kubectl logs -f <pod-name>

# View logs from all pods in a job at once
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<WORKLOAD_NAME>

# View logs from a specific pod index (e.g., worker pod 0)
kubectl logs <WORKLOAD_NAME>-worker-0-0-<suffix>
kubectl logs <WORKLOAD_NAME>-worker-0-1-<suffix>
```

Pod naming convention: `<JOBSET_NAME>-<JOB_NAME>-<JOB_INDEX>-<RANDOM_SUFFIX>`
- Example: `tpu-test-worker-0-0-s2tfh` (jobset=tpu-test, job=worker, index=0)
- Example: `tpu-test-worker-0-1-qh6w8` (jobset=tpu-test, job=worker, index=1)

### Clean up

```bash
kubectl delete jobset <WORKLOAD_NAME>
```

### DLRM training example (16-node, 64-chip)

Existing file: `~/dlrm-job.yaml` on `linjian-client`

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: dlrm-v2-1
  namespace: default
spec:
  replicatedJobs:
  - name: worker
    replicas: 1
    template:
      spec:
        parallelism: 16
        completions: 16
        template:
          spec:
            containers:
            - name: dlrm
              image: us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest
              command: ["python", "recml/examples/dlrm_experiment_test.py"]
              ports:
              - containerPort: 8471
              resources:
                limits:
                  google.com/tpu: 4
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 4x4x4
            restartPolicy: Never
```

### TPU test workload (verification)

Use this to verify a cluster's TPU functionality. File: `tpu-test-job.yaml`

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: tpu-test
  namespace: default
spec:
  replicatedJobs:
  - name: worker
    replicas: 1
    template:
      spec:
        parallelism: 2        # ← adjust to match your cluster's TPU node count
        completions: 2
        template:
          spec:
            containers:
            - name: tpu-test
              image: us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest
              command:
              - python3
              - -c
              - |
                import jax
                import jax.numpy as jnp
                print("=" * 60)
                print("TPU Multi-Node Test")
                print("=" * 60)
                devices = jax.devices()
                local_devices = jax.local_devices()
                print(f"Process index: {jax.process_index()}")
                print(f"Process count: {jax.process_count()}")
                print(f"Total devices: {len(devices)}")
                print(f"Local devices: {len(local_devices)}")
                for d in local_devices:
                    print(f"  - {d}")
                x = jnp.ones((1024, 1024))
                y = jnp.dot(x, x)
                print(f"\nTest 1 - Local matmul: shape={y.shape}, sum={float(y.sum()):.0f}")
                @jax.pmap
                def add_one(x):
                    return x + 1
                data = jnp.arange(len(local_devices), dtype=jnp.float32)
                result = add_one(data)
                print(f"Test 2 - pmap result: {result}")
                print("\n" + "=" * 60)
                print(f"SUCCESS: Process {jax.process_index()} completed all tests")
                print("=" * 60)
              ports:
              - containerPort: 8471
              resources:
                limits:
                  google.com/tpu: 4
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 2x2x1
            restartPolicy: Never
```

**Expected output** (both pods):
```
============================================================
TPU Multi-Node Test
============================================================
Process index: 0
Process count: 1
Total devices: 8
Local devices: 8
  - TPU_0(process=0,(0,0,0,0))
  - TPU_1(process=0,(0,0,0,1))
  ...
Test 1 - Local matmul: shape=(1024, 1024), sum=1073741824
Test 2 - pmap result: [1. 2. 3. 4. 5. 6. 7. 8.]
============================================================
SUCCESS: Process 0 completed all tests
============================================================
```

### TPU topology reference

| Nodes | Chips | Topology | Notes |
|---|---|---|---|
| 1 | 4 | `2x2x1` | Single-host |
| 2 | 8 | `2x2x1` | Multi-host, ICI connected |
| 4 | 16 | `4x4x4` | Multi-host, ICI connected |

> **Key insight:** Multi-node TPU slices are connected via ICI (not DCN). JAX sees
> all chips as a single unified device mesh with `process_count=1`. This means
> `jax.pmap` and `jax.sharding` automatically distribute across all physical nodes
> transparently — no manual distributed initialization needed.

### Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `admission webhook "warden-validating..." denied the request` | Missing `cloud.google.com/gke-tpu-topology` in nodeSelector | Add the topology label (check it with the topology command above) |
| `can't open file '...': No such file or directory` | Container image doesn't include source code (stale `:latest` tag) | Rebuild and push the image (see workflow below) |
| Pods `Error` immediately after start | Usually a missing file or import error | Check `kubectl logs <pod-name>` for the specific error |
| Pods stuck in `Pending` | Not enough TPU nodes, or wrong topology/accelerator label | Check `kubectl describe pod <name>` for scheduling errors |
| Pods stuck in `ContainerCreating` | Image pull in progress (first time is slow, ~1-2 min) | Wait, or check `kubectl describe pod <name>` for image pull errors |
| `RBAC` / `Forbidden` errors when installing JobSet | VM service account lacks cluster-admin | Run the command from your **local Mac** instead (authenticated as `linjianma@meta.com`) |
| `No matching distribution found for xpk` | VM Python is 3.9, XPK requires ≥3.10 | Use `gcloud` commands directly instead of XPK |
| CUDA errors in logs (`Unable to register cuFFT/cuBLAS`) | Harmless — TPU nodes don't have GPUs, JAX falls through to TPU backend | Ignore these warnings |

### ⚠️ Important: Image rebuild workflow

**Always rebuild and push the image before submitting jobs if code has changed.**
GKE nodes cache images aggressively — even with `imagePullPolicy: Always`, stale
`:latest` tags can cause failures if the registry image is outdated.

```bash
# On linjian-client:
cd ~/RecML

# 1. Rebuild
docker build -t recml-training .

# 2. Tag
docker tag recml-training:latest \
  us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest

# 3. Push
docker push \
  us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest

# 4. Verify the image has your code
docker run --rm \
  us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest \
  ls /app/recml/examples/
```

**Always use `imagePullPolicy: Always`** in your workload YAML to force GKE nodes
to pull the latest image on every run:

```yaml
containers:
- name: training
  image: us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:latest
  imagePullPolicy: Always    # ← prevents stale cached images on TPU nodes
```

Alternatively, tag images with a version instead of `:latest` to avoid caching issues:

```bash
export TAG=$(date +%Y%m%d-%H%M%S)
docker tag recml-training:latest \
  us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:$TAG
docker push \
  us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/recml_base_image:$TAG
```

---

## 4. Existing Clusters

### recir-v7-8-1

| Property | Value |
|---|---|
| **Kubectl context** | `gke_mrs-tpu-y317h_us-central1_recir-v7-8-1` |
| **Created** | 2026-04-11 |
| **K8s version** | 1.35.2-gke.1485000 |
| **CPU pool** | 6 nodes, `e2-standard-16`, autoscaling 1-1000 |
| **TPU pool** | `recir-v7-8-1-np-0`: 1 node, `tpu7x-standard-4t` (4 chips) |
| **TPU topology** | `2x2x1` |
| **Reservation** | `batch2pod2s` |
| **JobSet CRD** | installed |
| **Console** | https://console.cloud.google.com/kubernetes/workload_/gcloud/us-central1/recir-v7-8-1?project=mrs-tpu-y317h |

### recir-v7-8-2

| Property | Value |
|---|---|
| **Kubectl context** | `gke_mrs-tpu-y317h_us-central1_recir-v7-8-2` |
| **Created** | 2026-04-17 |
| **K8s version** | 1.35.2-gke.1485000 |
| **CPU pool** | 6 nodes, `e2-standard-16`, autoscaling 1-1000 |
| **TPU pool** | `recir-v7-8-2-np-0`: 2 nodes, `tpu7x-standard-4t` (8 chips total) |
| **TPU topology** | `2x2x1` |
| **Reservation** | `batch2pod2s` |
| **JobSet CRD** | installed |
| **Console** | https://console.cloud.google.com/kubernetes/workload_/gcloud/us-central1/recir-v7-8-2?project=mrs-tpu-y317h |

### Reservation: batch2pod2s

| Property | Value |
|---|---|
| **Location** | `us-central1-ai1a` |
| **TPU type** | TPU v7x |
| **Total chips** | 9,216 |
| **VM family** | `VM_FAMILY_CLOUD_TPU_POD_SLICE_TPU7X` |
| **Expires** | 2031-04-08 |
| **Status** | READY |

### Key files on linjian-client

| File | Purpose |
|---|---|
| `~/RecML/` | RecML training code (cloned from `github.com/AI-Hypercomputer/RecML`) |
| `~/RecML/Dockerfile` | Container image definition (Python 3.12, JAX, protobuf, jax-tpu-embedding) |
| `~/RecML/requirements.txt` | Python dependencies |
| `~/setup.sh` | One-time setup script (gke auth plugin, DNS access) |
| `~/train.sh` | Build & push image, then launch training via XPK |
| `~/dlrm-job.yaml` | DLRM training JobSet (16 nodes, 64 chips, topology `4x4x4`) |
