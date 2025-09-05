# NEX Backend - Healthcare Infection Detection System

A FastAPI-based healthcare analytics platform that detects infection clusters and tracks disease spread patterns using microbiology data and patient transfer records.

## Features

- **Infection Cluster Detection**: Identifies connected groups of infected patients using spatial-temporal contact analysis
- **Contact Tracing**: Tracks patient interactions based on location overlaps and timing
- **Risk Assessment**: Analyzes transmission patterns and identifies high-risk locations
- **AI-Powered Insights**: LLM-generated summaries and narrative analysis of infection patterns
- **Interactive Visualizations**: Patient networks, location heatmaps, and temporal spread patterns
- **RESTful API**: 12+ endpoints for accessing analytics data and visualizations

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nex-project/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On macOS/Linux
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # For production
   pip install -e .
   
   # For development (includes testing and code quality tools)
   pip install -e .[dev]
   ```

4. **Configure environment variables** (optional)
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here  # For AI summaries
   PROJECT_NAME="NEX Backend"
   ROOT_PATH=""
   ```

## Usage

### Starting the Server

**Production mode:**
```bash
python run.py
```

**Development mode (with hot reload):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`

### Data Requirements

The application expects two CSV files in the `app/data/` directory:

1. **microbiology.csv** - Test results with columns:
   - `test_id`: Unique test identifier
   - `patient_id`: Patient identifier
   - `collection_date`: Test collection date (YYYY-MM-DD)
   - `infection`: Infection type (CRE, ESBL, VRE, MRSA)
   - `result`: Test result (positive/negative)

2. **transfers.csv** - Patient movements with columns:
   - `transfer_id`: Unique transfer identifier
   - `patient_id`: Patient identifier
   - `ward_in_time`: Entry date (YYYY-MM-DD)
   - `ward_out_time`: Exit date (YYYY-MM-DD)
   - `location`: Location name

### API Endpoints

Access the interactive API documentation at `http://localhost:8080/docs`

**Key endpoints:**
- `GET /` - Health check
- `GET /api/v1/infection-detection/summary` - Get infection summary metrics
- `GET /api/v1/infection-detection/clusters` - Get detected infection clusters
- `GET /api/v1/infection-detection/patients` - Get all patient data
- `GET /api/v1/infection-detection/visualization/spread` - Get spread visualization data
- `GET /api/v1/infection-detection/super-spreaders` - Identify high-risk patients

### Example API Usage

```bash
# Get infection summary
curl http://localhost:8080/api/v1/infection-detection/summary

# Get all clusters
curl http://localhost:8080/api/v1/infection-detection/clusters

# Get specific patient data
curl http://localhost:8080/api/v1/infection-detection/patients/P0001
```

## Development

### Code Quality

```bash
# Run linting
ruff check .

# Format code
ruff format .

# Type checking
mypy app/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov

# Run specific test file
pytest tests/test_specific.py

# Generate HTML coverage report
pytest --cov-report=html
```

## Architecture

The system processes CSV data through a pipeline:

1. **Data Loading**: Reads microbiology and transfer CSV files
2. **Contact Detection**: Identifies patient interactions based on location/time overlaps
3. **Cluster Formation**: Groups connected patients using Union-Find algorithm
4. **Analytics Generation**: Produces metrics, visualizations, and AI summaries

**Core Components:**
- `InfectionDetectionService`: Main business logic for cluster detection
- `FastAPI Router`: RESTful API endpoints for data access
- `LLM Integration`: AI-powered analysis using LangChain + OpenAI

## Data Privacy & Security

- All patient data should be de-identified before processing
- The system works with patient IDs, not personally identifiable information
- Ensure compliance with healthcare data regulations (HIPAA, GDPR) in your jurisdiction

## Configuration

Settings are managed via `app/core/config.py` using Pydantic. Key configurations:
- Server host/port settings
- CORS origins
- File paths for CSV data
- Logging configuration

## Contributing

1. Install development dependencies: `pip install -e .[dev]`
2. Run code quality checks before committing
3. Ensure tests pass with minimum 80% coverage
4. Follow existing code style and patterns
