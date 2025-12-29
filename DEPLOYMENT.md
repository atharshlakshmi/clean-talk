# Google Cloud Deployment Guide

This guide walks you through deploying your Clean Talk app to Google Cloud.

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click "Select a Project" → "New Project"
3. Name it: `clean-talk`
4. Click "Create"
5. Wait for it to be created, then select it

## Step 2: Set Up Cloud Run (Hosting Service)

Cloud Run is Google's serverless container platform - perfect for your Docker setup.

```bash
# Enable the necessary APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable container.googleapis.com
```

## Step 3: Create an Artifact Registry Repository

This is where your Docker images are stored.

```bash
# Create a repository for your images
gcloud artifacts repositories create clean-talk \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for Clean Talk"
```

## Step 4: Create a Service Account (For GitHub to Deploy)

GitHub needs permission to deploy. Create a special account for this:

```bash
# Create service account
gcloud iam service-accounts create github-deployer \
  --display-name="GitHub Actions Deployer"

# Grant permissions to push images
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:github-deployer@$PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/artifactregistry.writer

# Grant permissions to deploy to Cloud Run
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:github-deployer@$PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/run.admin

# Grant permissions to pass service accounts
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:github-deployer@$PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/iam.serviceAccountUser
```

## Step 5: Set Up GitHub Secrets (For Secure Authentication)

1. Go to your GitHub repo
2. Settings → Secrets and variables → Actions
3. Add these secrets:

### New Secret 1: `GCP_PROJECT_ID`
- **Value**: Your Google Cloud Project ID (from Step 1)

### New Secret 2: `GCP_WORKLOAD_IDENTITY_PROVIDER`
- This enables secure GitHub→Google Cloud authentication without storing keys
- Run this command:

```bash
gcloud iam workload-identity-pools create "github" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub"

gcloud iam workload-identity-pools providers create-oidc "github" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.aud=assertion.aud,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Copy the output and save as GCP_WORKLOAD_IDENTITY_PROVIDER
gcloud iam workload-identity-pools describe "github" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --format='value(name)'
```

### New Secret 3: `GCP_SERVICE_ACCOUNT`
- **Value**: `github-deployer@YOUR-PROJECT-ID.iam.gserviceaccount.com`

Then map the service account to GitHub:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  "github-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --subject="repo:YOUR-GITHUB-USERNAME/clean-talk:ref:refs/heads/main"
```

## Step 6: Create Environment File for Production

Create `.env.cloud` in your repo root with production variables:

```
# Production API Keys
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env_here
```

**Do NOT commit this to GitHub!** Add to `.gitignore`:
```
.env
.env.cloud
```

## Step 7: Push to GitHub

```bash
git add .github/workflows/deploy-gcp.yml
git commit -m "Add Google Cloud deployment workflow"
git push origin main
```

Now every time you push to `main`, GitHub will:
1. ✅ Test your code
2. 🐳 Build Docker images
3. 📤 Upload to Google Cloud
4. 🚀 Deploy to Cloud Run

## Step 8: Monitor Deployments

1. Go to your GitHub repo → "Actions" tab
2. See each deployment's status
3. Once green, your app is live!

## Step 9: Get Your Live URLs

After first deployment, find your URLs:

```bash
# API URL
gcloud run services describe clean-talk-api \
  --region us-central1 \
  --format='value(status.url)'

# App URL
gcloud run services describe clean-talk-app \
  --region us-central1 \
  --format='value(status.url)'
```

## Useful Commands

```bash
# View logs from your app
gcloud run services logs read clean-talk-app --limit 50

# Update environment variables
gcloud run services update clean-talk-api \
  --update-env-vars KEY=VALUE \
  --region us-central1

# View all running services
gcloud run services list
```

## Estimated Costs

Google Cloud Free Tier includes:
- ✅ 2 million requests/month on Cloud Run
- ✅ 360,000 GB-seconds of compute/month
- ✅ Enough for a small project!

## Troubleshooting

**Deployment fails?**
- Check GitHub Actions logs for errors
- Verify secrets are set correctly
- Make sure APIs are enabled in GCP console

**Images not pushing?**
- Check Artifact Registry exists
- Verify service account has `artifactregistry.writer` role

**App crashes after deploy?**
- Check `gcloud run services logs read clean-talk-app`
- Verify `.env.cloud` has all required keys
- Check memory/CPU allocation in Cloud Run console

## Next Steps

- Monitor your app at: https://console.cloud.google.com/run
- Set up custom domain (DNS settings in GCP console)
- Enable Cloud CDN for faster content delivery
- Set up alerts for errors/latency
