# UnPointMaps - Data Mining Project

**Data Mining Project: Discover and describe areas of interest and events from geo-located data**

## Project Context

This project was developed as part of the Data Mining course at INSA Lyon. The goal is to help Grand Lyon improve public transport and tourist experiences by identifying areas of high tourist density using a cost-effective, non-intrusive approach based on geolocated Flickr photographs.

The project implements a complete knowledge discovery pipeline:
1. **Data Understanding & Preprocessing**: Analysis and cleaning of 400,000+ Flickr photo records
2. **Clustering**: Discovery of areas of interest using multiple algorithms
3. **Text Mining**: Description of clusters using photo metadata
4. **Temporal Analysis**: Identification of events and temporal patterns

## Project Objectives

### Primary Objectives
- **Area Discovery**: Automatically identify localized areas with high photo-taking activity
- **Algorithm Comparison**: Experiment with k-means, hierarchical clustering, DBSCAN, and HDBSCAN
- **Text Processing**: Use TF-IDF and association rules to describe areas of interest
- **Temporal Analysis**: Study how areas evolve over time and identify events

### Pedagogical Aims
- Implement techniques for handling large data collections
- Experiment with clustering algorithms and understand their parameters
- Apply text processing and natural language techniques
- Demonstrate scientific methodology and rigor in data mining choices

## Features

### Clustering Algorithms Implementation
- **K-Means**: Traditional centroid-based clustering with automatic optimal k selection (elbow/silhouette methods)
- **DBSCAN**: Density-based clustering for arbitrary-shaped clusters and noise detection
- **Iterative HDBSCAN**: Hierarchical clustering with automatic large cluster splitting
- **Parallel HDBSCAN**: Multi-threaded version for improved performance on large datasets

### Interactive Web Interface
- Real-time clustering visualization with WebSocket updates
- Interactive map with cluster boundaries using concave hulls
- Progress tracking for long-running clustering operations
- Cluster labeling with AI-generated descriptions

### AI-Powered Labeling
- LLM integration for automatic cluster description generation
- Semantic analysis of photo titles, tags, and descriptions
- Keyword extraction using TF-IDF and word frequency analysis
- Caching system for efficient re-processing

### Data Processing & Filtering
- Time-based filtering (years, hours, events)
- Dataset preprocessing and cleaning (duplicate removal, GPS validation)
- Support for large datasets with efficient caching
- Flickr API data integration

### Developer-Friendly Architecture
- RESTful API for integration
- Command-line interface for batch processing
- Comprehensive test suite
- Modular architecture following software engineering best practices

## Data Source

The project uses a dataset of over 400,000 geolocated Flickr photographs collected via the Flickr API. Each record contains:
- Photo ID and photographer ID
- GPS coordinates (latitude, longitude)
- Tags and description text
- Timestamp information

**Data Format**: `⟨id_photo, id_photographe, latitude, longitude, tags, description, dates⟩`

## Project Milestones & Evaluation

### Session 1 Milestones (5/20 points)
- ✅ Explore data and identify main problems
- ✅ Implement data cleaning (duplicates, GPS validation, etc.)
- ✅ Working visualization with map and data points
- ✅ First clustering algorithm implementation

### Session 2 Milestones (5/20 points)
- ✅ Complete data cleaning pipeline
- ✅ Implement and optimize 3 clustering algorithms
- ✅ Cluster visualization on map
- ✅ First text pattern mining algorithm

### Session 3 Milestones (10/20 points)
- ✅ Final clustering optimization and comparison
- ✅ Complete 2 text mining algorithms for cluster naming
- ✅ Temporal analysis and event detection
- ✅ Final demo preparation

## Installation

### Prerequisites
- Python 3.8+
- Node.js (for frontend linting)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/LouReinaK/UnPointMaps.git
   cd UnPointMaps
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies (optional, for linting)**
   ```bash
   npm install
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Web Interface (Primary Demonstration)

1. **Start the server**
   ```bash
   python server.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8000`

3. **Use the interface**
   - Select clustering algorithm and parameters
   - Apply time/event filters
   - Watch real-time clustering progress
   - Explore AI-labeled clusters on the interactive map

## Practical Applications for Grand Lyon

The clustering results provide actionable insights for urban planning:

- **Tourist Hotspots**: Identify areas needing improved public transport
- **Event Detection**: Discover recurring festivals and temporary attractions
- **Infrastructure Planning**: Optimize tourist facilities placement
- **Traffic Management**: Anticipate crowd-related congestion
- **Economic Development**: Support tourism industry growth strategies

## Clustering Algorithms Implementation

The project implements and compares four clustering algorithms as required by the course objectives:

### K-Means Clustering
- **Educational Focus**: Understanding centroid-based clustering and parameter optimization
- **Implementation**: Automatic optimal k selection using elbow and silhouette methods
- **Best for**: Well-separated, spherical clusters
- **Parameters Studied**: k value, initialization methods, convergence criteria
- **Pros**: Fast, deterministic, easy to understand
- **Cons**: Sensitive to initialization, assumes spherical clusters

### DBSCAN Clustering
- **Educational Focus**: Density-based clustering and noise handling
- **Implementation**: Optimized with kd-tree algorithm and parallel processing
- **Best for**: Arbitrary-shaped clusters with noise detection
- **Parameters Studied**: eps (neighborhood distance), min_samples
- **Pros**: Finds arbitrary shapes, handles noise, no need to specify k
- **Cons**: Sensitive to parameter tuning, struggles with varying densities

### Iterative HDBSCAN
- **Educational Focus**: Hierarchical clustering and large cluster handling
- **Implementation**: Automatic splitting of oversized clusters with intermediate results
- **Best for**: Large datasets with varying cluster densities
- **Parameters Studied**: min_cluster_size, cluster_selection_epsilon, max_cluster_size
- **Pros**: Handles different densities, provides hierarchical structure
- **Cons**: Computationally intensive, complex parameter space

### Parallel HDBSCAN
- **Educational Focus**: Performance optimization and parallel computing
- **Implementation**: Multi-threaded processing with ProcessPoolExecutor
- **Best for**: Maximum performance on multi-core systems
- **Parameters Studied**: Same as iterative HDBSCAN plus worker count
- **Pros**: Best performance on modern hardware
- **Cons**: Most complex implementation, resource intensive

### Algorithm Comparison & Selection
Based on our experiments with the Lyon Flickr dataset:
- **Recommended Algorithm**: [Your conclusion based on results]
- **Performance Metrics**: Silhouette scores, execution time, cluster quality
- **Parameter Sensitivity**: Analysis of how parameters affect results
- **Scalability**: Performance comparison on different dataset sizes

## Configuration

### Clustering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 10 | Minimum points for a valid cluster |
| `cluster_selection_epsilon` | 0.001° | Density threshold (≈100m) |
| `max_cluster_size` | 1000-5000 | Maximum cluster size before splitting |
| `eps` (DBSCAN) | 0.3 | Neighborhood distance |
| `min_samples` (DBSCAN) | 5 | Core point requirement |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | API key for LLM labeling | Yes |

## Text Mining Implementation

### Preprocessing Pipeline
- **Stop Word Removal**: English and French stop words using NLTK
- **Tokenization**: Word-level tokenization with stemming/lemmatization
- **Data Cleaning**: Removal of frequent non-meaningful words, duplicate handling

### TF-IDF Analysis
- **Term Frequency**: Word frequency within individual clusters
- **Inverse Document Frequency**: Comparison across all clusters
- **Keyword Extraction**: Top-N most representative words per cluster
- **Word Cloud Visualization**: Visual representation of cluster content

### Association Rules (Future Enhancement)
- **Itemset Mining**: Discovery of frequent word combinations
- **Rule Generation**: Identification of meaningful word associations
- **Confidence Metrics**: Statistical significance of discovered patterns

## Project Structure

```
UnPointMaps/
├── src/
│   ├── clustering/          # Clustering algorithm implementations
│   │   ├── clustering.py           # K-means with auto k-selection
│   │   ├── dbscan_clustering.py    # DBSCAN implementation
│   │   ├── hdbscan_clustering.py   # Iterative HDBSCAN
│   │   └── parallel_hdbscan_clustering.py  # Parallel HDBSCAN
│   ├── processing/          # Data processing modules
│   │   ├── dataset_filtering.py   # Data cleaning and validation
│   │   ├── embedding_service.py   # Text embeddings for AI labeling
│   │   ├── llm_labelling.py       # AI-powered cluster naming
│   │   ├── remove_nonsignificative_words.py  # Text preprocessing
│   │   └── time_filtering.py      # Temporal filtering
│   ├── utils/               # Utility functions
│   ├── visualization/       # Map generation and plotting
│   └── database/            # Caching and persistence
├── static/                  # Web frontend
│   ├── index.html          # Main interface
│   ├── app.js              # Frontend logic
│   └── style.css           # Styling
├── tests/                   # Comprehensive test suite
├── server.py                # FastAPI web server
├── run.py                   # CLI entry point
├── flickr_data2.csv         # Dataset (400k+ Flickr photos)
├── clustering_tools_report.md  # Technical algorithm analysis
└── requirements.txt         # Python dependencies
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_clustering.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Python linting
python -m pylint src/ tests/ server.py

# JavaScript linting
npx eslint static/app.js

# CSS linting
npx stylelint static/style.css
```

### AI Usage in Development

As per course guidelines, AI was used responsibly for:
- Understanding complex algorithms and libraries
- Code structure suggestions and debugging assistance
- Learning new concepts and best practices
- All AI-generated code was reviewed, understood, and properly documented

### Algorithm Implementation Guidelines

Following the course requirement to implement algorithms from scratch where possible:
- Core clustering logic implemented using scikit-learn as the foundation
- Custom parameter optimization and result processing
- Extensive experimentation with different parameter combinations
- Performance benchmarking and algorithm comparison

### Adding New Clustering Algorithms

1. Create a new module in `src/clustering/`
2. Implement the clustering function following the established pattern
3. Add the algorithm to the server's algorithm selection logic
4. Update tests and documentation

## API Reference

### Endpoints

#### `GET /api/stats`
Returns dataset statistics and detected events.

**Response:**
```json
{
  "total_points": 10000,
  "min_date": "2004-01-01T00:00:00",
  "max_date": "2023-12-31T23:59:59",
  "events": [...]
}
```

#### `POST /api/cluster`
Initiates clustering with specified parameters.

**Request Body:**
```json
{
  "algorithm": "hdbscan|parallel_hdbscan|kmeans|dbscan",
  "min_year": 2010,
  "max_year": 2020,
  "start_hour": 8,
  "end_hour": 18,
  "exclude_events": [1, 2],
  "labelling_method": "llm|statistical"
}
```

#### WebSocket `/ws`
Real-time updates for clustering progress and results.

**Messages:**
- `{"type": "progress", "message": "...", "iteration": 1}`
- `{"type": "cluster_update", "clusters": [...]}`
- `{"type": "label_update", "cluster_id": 1, "label": "..."}`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Add tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, scikit-learn, HDBSCAN, and other open-source libraries
- Uses OpenAI API for LLM-powered labeling
- Inspired by geospatial analysis and clustering research

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the clustering_tools_report.md for detailed technical documentation
- Review the test suite for usage examples