pipeline {
  agent any
  environment {
    // Dummy secrets so import-time Settings()/clients don't crash during CI
    GOOGLE_API_KEY='ci-dummy'
    PINECONE_API_KEY='ci-dummy'
    HUGGINGFACE_TOKEN='ci-dummy'
    AWS_REGION='us-east-1'
    S3_BUCKET_NAME='ci-dummy-bucket'
    AWS_ACCESS_KEY_ID='ci-dummy'
    AWS_SECRET_ACCESS_KEY='ci-dummy'
    PINECONE_INDEX_NAME='rag-knowledge-base'
    LOG_JSON='true'
    UV_CACHE_DIR="${WORKSPACE}/.uv-cache"
  }
  options {
    timestamps()
    timeout(time: 20, unit: 'MINUTES')
  }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Setup uv') {
      steps {
        sh '''
          python3.12 --version
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="$HOME/.local/bin:$PATH"
          uv --version
        '''
      }
    }
    stage('Install') {
      steps {
        sh 'export PATH="$HOME/.local/bin:$PATH"; uv sync --frozen'
      }
    }
    stage('Lint') {
      steps {
        sh '''
          export PATH="$HOME/.local/bin:$PATH"
          uv run ruff check .
          uv run ruff format --check .
        '''
      }
    }
    stage('Type-check') {
      steps {
        sh '''
          export PATH="$HOME/.local/bin:$PATH"
          uv run mypy app.py config.py dependencies.py exceptions.py logging_config.py components database integrations
        '''
      }
    }
    stage('Test') {
      steps {
        // Coverage baseline: 54 (Phase 1 measured 2026-05-30; ratchet upward only)
        sh '''
          export PATH="$HOME/.local/bin:$PATH"
          uv run pytest --cov --cov-report=xml --junitxml=junit.xml --cov-fail-under=54
        '''
      }
    }
  }
  post {
    always {
      junit allowEmptyResults: true, testResults: 'junit.xml'
    }
  }
}
