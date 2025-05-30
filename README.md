# 📘 DocQueryEngine

A document ingestion and Retrieval-Augmented Generation (RAG)-based Q&A backend built with FastAPI, PostgreSQL + pgvector, Hugging Face Transformers, Langchain, and spaCy.

---

## 🚀 Features

- ✅ Upload `.pdf` and `.txt` files
- ✅ Extract and embed document content
- ✅ Ask questions against specific documents
- ✅ JWT-based authentication
- ✅ Postgres + pgvector storage
- ✅ FAISS retrieval support
- ✅ Dockerized with Gunicorn for production
- ✅ Environment-based config support

---

## 📁 Project Structure

```
.
├── src/
│   ├── api/                  # FastAPI routes
│   ├── core/                 # Embedding, retrieval, Q&A logic
│   ├── exception/            # Exception handler
│   ├── middleware/           # Middlewares for the application
│   ├── models/               # SQLAlchemy models
│   ├── service/              # Service layer 
│   ├── util/                 # Utlities for error, logging etc.
├── tests/                    # Containe unit tests for some endpoints and services
│   ├── api/
│   ├── service/
├── Dockerfile
├── docker-compose.yml
├── .env.dev
├── .env.prod
├── requirements.txt
└── README.md
```

---

## 🐳 Running with Docker

### Run PostgreSQL with pgvector using Docker Compose

Create a `docker-compose.yml` file like this:

```yaml
version: "3.8"

services:
  postgres:
    image: ankane/pgvector:latest
    container_name: rag_pgvector
    restart: always
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ragdb
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Then run:

```bash
docker-compose up -d
```

This will start the PostgreSQL database with pgvector extension enabled, ready to accept connections on `localhost:5432`.

---

### Build and run FastAPI container

```bash
docker build -t doc-query-engine .
docker run --rm -p 8000:8000 --network rag-network doc-query-engine
```

---

## ⚙️ Environment Variables

Environment-specific files:

- `.env.dev` → used for local development
- `.env.prod` → used inside Docker container

Example:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb
SECRET_KEY=supersecret
EMBEDDING_DIM=1024
```

---

## 🧪 Run Locally with Uvicorn

```bash
uvicorn main:app --reload
```

Ensure your `.env.dev` is active and database is running locally or via Docker.

---

## 🔐 Auth Endpoints

- `POST /api/v1/register`
- `POST /api/v1/login`

Uses access and refresh JWTs. Pass `Authorization: Bearer <access_token>` in secure endpoints.

---

## 🧐 Document Endpoints

- `POST /api/v1/ingest` → Ingest document (auth required)
- `POST /api/v1//list-documents` → Listing all uploaded documents by user
- `POST /api/v1/qa` → Ask question using `document_id`

---

## 🚀 Deployment to AWS ECR

### 1. Authenticate Docker with AWS

```bash
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
```

---

### 2. Create ECR Repository (if not created)

```bash
aws ecr create-repository --repository-name doc-query-engine --region <your-region>
```

---

### 3. Build and Tag Docker Image

```bash
docker build -t doc-query-engine .
docker tag doc-query-engine:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/doc-query-engine:latest
```

---

### 4. Push to ECR

```bash
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/doc-query-engine:latest
```

---

### 5. Deploy via ECS, Fargate, or EC2

You can now pull and deploy the container using:

- **ECS with Fargate**
- **EC2** (`docker run`)
- **Elastic Beanstalk** (multi-container)
- **EKS (Kubernetes)** with custom Helm chart

---

## 📉 GitHub CI/CD Pipeline Setup

### 1. Create `.github/workflows/ci.yml`

```yaml
name: CI Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest

  docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Log in to Amazon ECR
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build and push Docker image
        env:
          ECR_REGISTRY: <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
          ECR_REPOSITORY: doc-query-engine
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

### 2. Store secrets in GitHub repo settings
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

---

## 🛠 Design Patterns Used

- **Service Layer** – business logic in `service/`
- **Dependency Injection** – via `Depends()`
- **Repository Pattern** – DB queries abstracted via SQLAlchemy
- **Strategy Pattern** – supports FAISS and pgvector as interchangeable retrievers

---

## Enhancements

- Adding validations for limiting document upload size
- Ability to accept multiple documents for ingestion
- Ability to ingest multiple documents with Background Tasks 
- Maintaining a chat-history upto last k-number of chats
- Use of Redis with Celery for efficient cache management
- Role based authorization for users
- - Maintaining user-profiles for verification and activation.
- Maintaining token limit for users for a certain amount of time
- Maintain and training separate ML models
- Adding support for other documents like .docx, .json etc.
 

---

## ✅ License

MIT License
