# AI GenAI Auto-Correction Pipeline

A Python-based pipeline that captures LLM query-response pairs, compares them against ideal (ground-truth) responses, and automatically generates correction files for responses that fall below a configurable confidence threshold. These correction files are structured for future model training and fine-tuning.

## Objectives

- Understand how LLM self-correction pipelines detect and capture low-quality responses using configurable confidence thresholds
- Implement validation loops that compare generated outputs against ground-truth references to measure response quality
- Apply the Strategy pattern to swap between comparison methods (cosine similarity vs. LLM-as-judge) without modifying pipeline logic
- Use the Factory pattern to abstract LLM provider selection, enabling seamless switching between OpenAI, Anthropic, and Azure OpenAI
- Build structured correction datasets in markdown format suitable for future model fine-tuning and training workflows
- Design async Python pipelines with clean separation of concerns across configuration, providers, comparison, and correction modules
- Leverage embedding-based semantic similarity (cosine similarity) to quantify how closely an LLM response matches an ideal answer
- Implement an LLM-as-judge evaluation pattern where one model scores another model's output with an explanation
- Apply production-grade practices including immutable configuration, YAML-driven settings, rotating log files, and environment-based secrets management
- Structure a Python project with Poetry, strict type checking (mypy), linting (ruff), security scanning (bandit), and high test coverage requirements

## How It Works

1. **Query Processing**: Sends queries to a configured LLM provider (OpenAI, Anthropic, or Azure OpenAI)
2. **Response Comparison**: Compares the generated response against an ideal response using either:
   - **Cosine Similarity**: Embedding-based semantic comparison
   - **LLM-as-Judge**: Uses an LLM to evaluate response quality
3. **Correction Capture**: If the confidence score falls below the threshold (default: 0.8), the query, actual response, and ideal response are saved to a structured markdown file for future correction/training

## Project Structure

```
ai-genai-autocorrection-pipeline/
├── .gitignore                          # Git ignore patterns for Python projects
├── .env.example                        # Template for environment variables
├── README.md                           # This file
├── pyproject.toml                      # Poetry project configuration and dependencies
├── config/
│   ├── settings.yaml                   # Main application configuration
│   ├── logging.yaml                    # Logging configuration (levels, handlers)
│   └── ideal_responses.json            # Query-ideal response pairs (ground truth)
├── src/
│   └── autocorrection_pipeline/
│       ├── __init__.py                 # Package initialization
│       ├── main.py                     # CLI entry point and bootstrap
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py            # YAML + env var configuration loader
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py                # Abstract LLM provider interface
│       │   ├── factory.py             # Factory pattern for provider creation
│       │   ├── openai_provider.py     # OpenAI API implementation
│       │   ├── anthropic_provider.py  # Anthropic Claude implementation
│       │   └── azure_openai_provider.py # Azure OpenAI implementation
│       ├── comparison/
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract comparator interface
│       │   ├── cosine_similarity.py   # Embedding-based comparison
│       │   └── llm_judge.py          # LLM-as-judge comparison
│       ├── correction/
│       │   ├── __init__.py
│       │   └── capture.py            # Markdown correction file generator
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── orchestrator.py       # Main pipeline coordination
│       ├── logging_config/
│       │   ├── __init__.py
│       │   └── setup.py              # Logging initialization from YAML
│       └── utils/
│           ├── __init__.py
│           └── validators.py          # Input validation and sanitization
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared pytest fixtures
│   ├── unit/                          # Unit tests (one per module)
│   │   ├── test_settings.py
│   │   ├── test_openai_provider.py
│   │   ├── test_anthropic_provider.py
│   │   ├── test_azure_openai_provider.py
│   │   ├── test_factory.py
│   │   ├── test_cosine_similarity.py
│   │   ├── test_llm_judge.py
│   │   ├── test_capture.py
│   │   ├── test_orchestrator.py
│   │   └── test_validators.py
│   └── integration/                   # Integration tests
│       └── test_pipeline_integration.py
├── output/                            # Generated correction markdown files
│   └── .gitkeep
└── logs/                              # Application log files
    └── .gitkeep
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ^1.82.0 | OpenAI and Azure OpenAI API client |
| `anthropic` | ^0.52.0 | Anthropic Claude API client |
| `numpy` | ^2.2.0 | Vector math for cosine similarity computation |
| `pyyaml` | ^6.0.2 | YAML configuration file parsing |
| `python-dotenv` | ^1.1.0 | Environment variable loading from .env files |
| `pydantic` | ^2.11.0 | Data validation support |

### Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ^8.3.0 | Test framework |
| `pytest-asyncio` | ^0.26.0 | Async test support |
| `pytest-cov` | ^6.1.0 | Test coverage reporting |
| `pytest-mock` | ^3.14.0 | Mocking utilities for tests |
| `ruff` | ^0.11.0 | Linting and code formatting |
| `mypy` | ^1.15.0 | Static type checking |
| `bandit` | ^1.9.0 | Security vulnerability scanning |

## Deployment

### Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/) package manager (1.8+)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-genai-autocorrection-pipeline
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-your-actual-key
   ANTHROPIC_API_KEY=sk-ant-your-actual-key
   AZURE_OPENAI_API_KEY=your-azure-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   ```

4. **Configure the pipeline** (optional):
   Edit `config/settings.yaml` to adjust:
   - `llm.active_provider`: Switch between `openai`, `anthropic`, or `azure_openai`
   - `llm.confidence_threshold`: Adjust the correction capture threshold (default: 0.8)
   - `llm.comparison_method`: Choose `cosine_similarity` or `llm_judge`
   - Provider-specific model settings

5. **Add ideal responses**:
   Edit `config/ideal_responses.json` with your query-response pairs:
   ```json
   {
     "queries": [
       {
         "query": "Your question here",
         "ideal_response": "The ideal answer here"
       }
     ]
   }
   ```

### Running the Pipeline

```bash
poetry run autocorrect
```

This processes all queries in `config/ideal_responses.json`, generates LLM responses, compares them against ideals, and writes correction files to `output/` for any responses below the confidence threshold.

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run only unit tests
poetry run pytest tests/unit/

# Run only integration tests
poetry run pytest tests/integration/
```

### Linting and Type Checking

```bash
# Run linter
poetry run ruff check src/ tests/

# Auto-fix lint issues
poetry run ruff check src/ tests/ --fix

# Run type checker
poetry run mypy src/

# Run security scanner
poetry run bandit -r src/
```

## Configuration

### Settings File (`config/settings.yaml`)

All pipeline behavior is configurable via this YAML file:

- **Provider selection**: Switch LLM providers without code changes
- **Confidence threshold**: Control when corrections are captured
- **Comparison method**: Choose between embedding similarity or LLM judge
- **Model parameters**: Adjust temperature, max tokens, and model versions

### Logging (`config/logging.yaml`)

Supports configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL):

- **Console handler**: Standard output for development
- **File handler**: Rotating file handler (10MB max, 5 backups) for production
- **Override**: Set `LOG_LEVEL` environment variable to override at runtime

### Environment Variables

All sensitive values (API keys) are loaded from environment variables via `.env`:

| Variable | Required For | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | OpenAI provider | OpenAI API authentication key |
| `ANTHROPIC_API_KEY` | Anthropic provider | Anthropic API authentication key |
| `AZURE_OPENAI_API_KEY` | Azure provider | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure provider | Azure OpenAI service endpoint URL |
| `LOG_LEVEL` | Optional | Override log level (DEBUG/INFO/WARNING/ERROR) |

## Output Format

Correction files are generated as markdown in the `output/` directory:

```markdown
# Correction Record

## Metadata
- **Timestamp**: 2024-01-15T10:30:00+00:00
- **Confidence Score**: 0.6200
- **Comparison Method**: cosine_similarity
- **Explanation**: Cosine similarity score: 0.6200...

## Query
What is the capital of France?

## Actual Response (LLM Generated)
Paris is a city in France...

## Ideal Response (Ground Truth)
The capital of France is Paris. It is the largest city...

---
*Captured for model training/correction purposes*
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI GenAI Auto-Correction Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  poetry run      │
                         │  autocorrect     │
                         └────────┬─────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BOOTSTRAP                                                           │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │ .env        │   │ settings.yaml    │   │ logging.yaml         │  │
│  │ (API keys)  │──▶│ (config)         │──▶│ (log config)         │  │
│  └─────────────┘   └──────────────────┘   └──────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│                     ┌──────────────────┐                             │
│                     │  AppSettings     │                             │
│                     │  (immutable)     │                             │
│                     └────────┬─────────┘                             │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INITIALIZE PIPELINE                                                 │
│                                                                      │
│  ┌────────────────────┐   ┌──────────────────┐  ┌───────────────┐  │
│  │  ProviderFactory   │   │  Comparator      │  │ Correction    │  │
│  │  ┌──────────────┐  │   │  ┌────────────┐  │  │ Capture       │  │
│  │  │ OpenAI       │  │   │  │ Cosine     │  │  │               │  │
│  │  │ Anthropic    │  │   │  │ Similarity │  │  │ (writes .md   │  │
│  │  │ Azure OpenAI │  │   │  │ LLM Judge  │  │  │  files)       │  │
│  │  └──────────────┘  │   │  └────────────┘  │  └───────────────┘  │
│  └────────────────────┘   └──────────────────┘                      │
│                                                                      │
│  ┌─────────────────────────────┐                                    │
│  │  ideal_responses.json       │                                    │
│  │  { query → ideal_response } │                                    │
│  └─────────────────────────────┘                                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BATCH PROCESSING (for each query)                                   │
│                                                                      │
│  ┌─────────┐      ┌───────────────┐      ┌──────────────────────┐  │
│  │  Query  │─────▶│  LLM Provider │─────▶│  Actual Response     │  │
│  └─────────┘      └───────────────┘      └──────────┬───────────┘  │
│                                                      │              │
│                    ┌───────────────┐                  │              │
│                    │ Ideal Response│◀─ (from JSON)    │              │
│                    └───────┬───────┘                  │              │
│                            │                         │              │
│                            ▼                         ▼              │
│                   ┌─────────────────────────────────────┐           │
│                   │         COMPARATOR                   │           │
│                   │                                     │           │
│                   │  Strategy A: Cosine Similarity      │           │
│                   │  ┌────────┐  ┌────────┐            │           │
│                   │  │Embed(A)│  │Embed(I)│→ cos(θ)    │           │
│                   │  └────────┘  └────────┘            │           │
│                   │                                     │           │
│                   │  Strategy B: LLM Judge              │           │
│                   │  ┌─────────────────────────┐       │           │
│                   │  │ LLM evaluates A vs I    │       │           │
│                   │  │ Returns {score, reason} │       │           │
│                   │  └─────────────────────────┘       │           │
│                   └──────────────┬──────────────────────┘           │
│                                  │                                   │
│                                  ▼                                   │
│                   ┌──────────────────────────┐                      │
│                   │  Confidence Score (0-1)  │                      │
│                   └────────────┬─────────────┘                      │
│                                │                                     │
│                    ┌───────────┴───────────┐                        │
│                    │                       │                        │
│              score >= threshold      score < threshold              │
│                    │                       │                        │
│                    ▼                       ▼                        │
│              ┌──────────┐     ┌────────────────────────┐           │
│              │  PASS    │     │  CORRECTION CAPTURE    │           │
│              └──────────┘     │  -> Write .md file     │           │
│                               │     to output/         │           │
│                               └────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  REPORT                                                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Summary: Total | Passed | Captured | Pass Rate %          │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### End-to-End Flow

1. **CLI Entry** (`main.py`): User runs `poetry run autocorrect`. Loads configuration from `config/settings.yaml`, API keys from `.env`, and logging from `config/logging.yaml`. Produces an immutable `AppSettings` dataclass.

2. **Pipeline Initialization** (`orchestrator.py`):
   - `ProviderFactory` selects the active LLM provider (OpenAI, Anthropic, or Azure OpenAI)
   - Instantiates the comparator (`CosineSimilarityComparator` or `LLMJudgeComparator`)
   - Sets up `CorrectionCapture` pointing to the `output/` directory
   - Loads ground-truth query/response pairs from `config/ideal_responses.json`

3. **Batch Processing** (per query):
   - **Generate**: Send query to configured LLM provider, get actual response
   - **Compare**: Evaluate actual vs. ideal using the selected strategy:
     - *Cosine Similarity*: Embed both responses, compute vector cosine similarity (0.0-1.0)
     - *LLM Judge*: Ask an LLM to score quality, parse JSON response with score + explanation
   - **Threshold Check**: If score < `confidence_threshold`, trigger correction capture
   - **Correction Capture**: Write structured markdown file with query, actual response, ideal response, and comparison metadata

4. **Reporting**: Log summary statistics (total processed, passed, captured, pass rate %) and exit.

### Design Patterns

The project follows these design patterns:

- **Factory Pattern**: `ProviderFactory` creates providers based on configuration
- **Strategy Pattern**: Comparison methods are interchangeable via `BaseComparator`
- **Immutable Data**: All configuration and result types are frozen dataclasses
- **Async/Await**: All I/O operations are asynchronous for efficiency
- **Single Responsibility**: Each class/module has one well-defined purpose

## Security

- API keys are stored exclusively in `.env` (git-ignored) and loaded via `python-dotenv`
- API key fields use `repr=False` to prevent accidental logging of secrets
- All external inputs are validated through `validators.py` before processing
- `bandit` is included for static security analysis of source code
- Output filenames are generated deterministically (never from user input)
