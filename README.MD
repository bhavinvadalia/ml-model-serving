```
  ██████╗ ██████╗ ███╗   ██╗████████╗ █████╗ ██╗███╗   ██╗███████╗██████╗
 ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██║████╗  ██║██╔════╝██╔══██╗
 ██║     ██║   ██║██╔██╗ ██║   ██║   ███████║██║██╔██╗ ██║█████╗  ██████╔╝
 ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██║██║██║╚██╗██║██╔══╝  ██╔══██╗
 ╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║██║██║ ╚████║███████╗██║  ██║
  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝

                                                                    ship ml models like you mean it 🚀
```

# 🛸 ml-model-rocket — CI/CD Pipeline for ML Model Serving

> A production-grade CI/CD pipeline that builds, tests, and deploys a FastAPI ML model to Kubernetes — fully automated with GitHub Actions.

```
 [Local Code]
      │
      │  git push
      ▼
 [GitHub Actions]
      │
      ├──► Build Docker Image
      │         │
      │         ▼
      │    Push to ghcr.io
      │
      ├──► Run Tests (pytest)
      │
      └──► Deploy to Kubernetes (k3s)
                │
                ├──► kubectl set image
                ├──► Wait for rollout
                ├──► Smoke Tests
                └──► Auto Rollback on Failure
```

---

## 🧰 Stack

| Layer | Tool |
|-------|------|
| App | FastAPI (Python) |
| Container | Docker |
| Registry | GitHub Container Registry (ghcr.io) |
| Kubernetes | k3s (self-hosted on Hetzner VPS) |
| CI/CD | GitHub Actions |
| Tests | pytest |

---

## 📋 Prerequisites

- A Linux VPS (Hetzner, DigitalOcean, etc.) with 2GB+ RAM
- GitHub account
- Docker installed locally
- `kubectl` installed locally

---

## 🚀 Step-by-Step Setup

### Step 1 — Install k3s on Your VPS

SSH into your server and run:

```bash
curl -sfL https://get.k3s.io | sh -
```

Verify it's running:

```bash
sudo kubectl get nodes
# NAME        STATUS   ROLES           AGE   VERSION
# your-server Ready    control-plane   30s   v1.x.x
```

---

### Step 2 — Open Required Ports

In your VPS provider's firewall (e.g. Hetzner console), open these inbound ports:

| Port | Purpose |
|------|---------|
| 22 | SSH |
| 6443 | Kubernetes API (GitHub Actions → k3s) |
| 30419 | App NodePort (smoke tests) |

> ⚠️ Make sure the firewall is **applied to your server** — not just created.

---

### Step 3 — Create Kubernetes Manifests

Create `kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-model-serving
  template:
    metadata:
      labels:
        app: ml-model-serving
    spec:
      imagePullSecrets:
      - name: ghcr-secret
      containers:
      - name: model-serving
        image: ghcr.io/<your-username>/ml-model-serving:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
```

Create `kubernetes/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-serving
spec:
  selector:
    app: ml-model-serving
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

### Step 4 — Deploy Manifests to k3s Locally First

Clone your repo on the VPS and apply:

```bash
git clone https://github.com/<your-username>/ml-model-serving.git
cd ml-model-serving
sudo kubectl apply -f kubernetes/deployment.yaml
sudo kubectl apply -f kubernetes/service.yaml
```

Create the image pull secret for ghcr.io:

```bash
sudo kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<your-github-username> \
  --docker-password=<your-github-pat> \
  --docker-email=noreply@github.com
```

Verify everything is running:

```bash
sudo kubectl get pods
sudo kubectl get services
```

Test your app:

```bash
curl http://<your-vps-ip>:30419/health
```

---

### Step 5 — Get kubeconfig for GitHub Actions

On your VPS, get the kubeconfig with your public IP substituted:

```bash
PUBLIC_IP=$(curl -s ipv4.icanhazip.com)
sudo cat /etc/rancher/k3s/k3s.yaml | sed "s/127.0.0.1/$PUBLIC_IP/g" | base64 -w 0
```

Copy the output.

---

### Step 6 — Add GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|-------------|-------|
| `KUBE_CONFIG` | base64 output from Step 5 |

---

### Step 7 — GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy ML Model

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-model-serving

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./docker/Dockerfile
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  deploy:
    runs-on: ubuntu-latest
    needs: [build, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          mkdir -p $HOME/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 --decode > $HOME/.kube/config
          chmod 600 $HOME/.kube/config

      - name: Create registry secret
        run: |
          kubectl create secret docker-registry ghcr-secret \
            --docker-server=ghcr.io \
            --docker-username=${{ github.actor }} \
            --docker-password=${{ secrets.GITHUB_TOKEN }} \
            --docker-email=noreply@github.com \
            -n default --dry-run=client -o yaml | kubectl apply -f -

      - name: Deploy
        run: |
          kubectl set image deployment/ml-model-serving \
            model-serving=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

      - name: Wait for rollout
        run: kubectl rollout status deployment/ml-model-serving --timeout=10m

      - name: Wait for service to stabilise
        run: sleep 30

      - name: Smoke tests
        run: |
          curl -f http://<your-vps-ip>:30419/health || exit 1
          curl -f -X POST http://<your-vps-ip>:30419/predict \
            -H "Content-Type: application/json" \
            -d '{"features": [5.1, 3.5, 1.4, 0.2]}' || exit 1

      - name: Rollback on failure
        if: failure()
        run: |
          kubectl rollout undo deployment/ml-model-serving
          kubectl rollout status deployment/ml-model-serving
```

---

### Step 8 — Push and Watch It Fly 🚀

```bash
git add .
git commit -m "🚀 initial ci/cd pipeline"
git push origin main
```

Go to your repo → **Actions tab** and watch the pipeline run.

```
  build ──► test ──► deploy
    ✅         ✅        ✅
```

---

## 🔁 How Rollback Works

If the smoke tests fail after a deployment, the pipeline automatically runs:

```bash
kubectl rollout undo deployment/ml-model-serving
```

This reverts to the previous working image with zero manual intervention.

---

## 📁 Project Structure

```
ml-model-serving/
├── .github/
│   └── workflows/
│       └── deploy.yml        # CI/CD pipeline
├── docker/
│   └── Dockerfile
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── src/
│   └── app/
├── tests/
├── requirements.txt
└── README.md
```

---

## 💡 Tips

- **k3s on a Hetzner CX21 (4GB)** is the sweet spot — cheap, fast, reliable
- **NodePort 30419** is auto-assigned — check with `kubectl get services` if it changes
- **3 replicas is too many** for a small VPS — use `replicas: 1` for dev/staging
- **Probe tuning matters** — give your ML model enough `initialDelaySeconds` to load

---

*Built with ☕ and a lot of kubectl debugging*
