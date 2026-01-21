# LLM Labeling System - Environment Configuration and Core Functionality
# Based on the architecture specified in plans/llm_labelling_architecture.md

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, ValidationError, ConfigDict
from dotenv import load_dotenv
import abc
import hashlib  # Added for caching


from src.database.manager import DatabaseManager  # Added for caching
# For Representative Sampling
from src.processing.embedding_service import EmbeddingService

# Global Rate Limiter Configuration
MAX_REQUESTS_PER_MINUTE = 1000  # Adjust this value to change the global rate limit


class RateLimiter:
    """Thread-safe rate limiter using a sliding window or simple delay mechanism."""

    def __init__(self, max_per_minute: int):
        self.limit = max_per_minute
        self.delay = 60.0 / max_per_minute
        self.last_call = 0.0
        self.lock = Lock()

    def update_limit(self, max_per_minute: int):
        """Dynamically update the rate limit."""
        with self.lock:
            if max_per_minute <= 0:
                max_per_minute = 1  # Avoid division by zero
            self.limit = max_per_minute
            self.delay = 60.0 / max_per_minute
            logger.info(
                f"Rate limit updated to {max_per_minute} requests/minute")

    def wait(self):
        """Blocks until a request can be made."""
        with self.lock:
            now = time.time()
            # Calculate when the next request is allowed
            next_allowed_time = self.last_call + self.delay
            wait_time = next_allowed_time - now

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_call = time.time()  # Update to now (after sleep)


# Initialize global rate limiter instance
_global_rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)


# Configure logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_labelling.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('llm_labelling')

# Custom Exception Classes


class LLMError(Exception):
    """Base class for LLM-related errors"""
    pass


class ConfigurationError(LLMError):
    """Errors related to system configuration"""
    pass


class InputValidationError(LLMError):
    """Errors related to input data validation"""
    pass


class APICommunicationError(LLMError):
    """Errors related to API communication"""
    pass


class LLMProcessingError(LLMError):
    """Errors related to LLM processing"""
    pass


class PostProcessingError(LLMError):
    """Errors related to label post-processing"""
    pass


class ConfigManager:
    """
    Configuration Manager for LLM Labeling System

    Handles loading, validation, and management of environment configuration
    including API keys, model settings, and operational parameters.
    """

    def __init__(self, env_path: str = '.env'):
        """
        Initialize ConfigManager with environment file path.

        Args:
            env_path: Path to .env file (default: '.env')
        """
        self.env_path = env_path
        self.config: Optional[Dict[str, Any]] = None
        self._validate_env_file()
        self._load_environment()

    def _validate_env_file(self) -> None:
        """
        Validate that the environment file exists and is readable.

        Raises:
            ConfigurationError: If .env file is missing or not readable
        """
        if not os.path.exists(self.env_path):
            raise ConfigurationError(
                f"Environment file not found: {self.env_path}")

        if not os.access(self.env_path, os.R_OK):
            raise ConfigurationError(
                f"Cannot read environment file: {self.env_path}")

    def _load_environment(self) -> None:
        """
        Load environment variables from .env file and validate required settings.

        Raises:
            ConfigurationError: If required environment variables are missing or invalid
        """
        try:
            # Load environment variables from .env file
            # Use override=True to ensure .env changes are picked up even if
            # env vars exist
            load_dotenv(self.env_path, override=True)

            # Identify provider
            provider = os.getenv('LLM_PROVIDER', 'openrouter').lower()

            # Validate model format
            model = os.getenv('DEFAULT_LLM_MODEL',
                              'mistralai/mistral-7b-instruct')
            if provider != 'ollama' and '/' not in model:
                raise ConfigurationError(
                    f"DEFAULT_LLM_MODEL must be in format provider/model when using OpenRouter. Got: {model}")

            timeout = self._parse_int_env('API_TIMEOUT', 30)
            if timeout <= 0:
                raise ConfigurationError(
                    f"API_TIMEOUT must be a positive integer, got: {timeout}")

            # Generic OpenAI Configuration
            base_url = os.getenv('OPENAI_BASE_URL')
            api_key = os.getenv('OPENAI_API_KEY')

            # Provider-specific defaults
            if not base_url:
                if provider == 'openrouter':
                    base_url = "https://openrouter.ai/api/v1"
                else:
                    # Default to Ollama/Localhost
                    base_url = "http://localhost:11434/v1"

            # Validation
            if provider == 'openrouter':
                if not api_key or not api_key.strip():
                    raise ConfigurationError(
                        "API key not found. Please set OPENAI_API_KEY for OpenRouter.")

            # Ensure API key is string
            if not api_key:
                api_key = "ollama"

            config = {
                'provider': provider,
                'model': model,
                'timeout': timeout,
                'max_retries': self._parse_int_env('MAX_RETRIES', 3),
                'env_path': self.env_path,
                'api_key': api_key.strip(),
                'base_url': base_url
            }

            self.config = config
            logger.info(
                f"Configuration loaded successfully from {self.env_path} (Provider: {provider})")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load environment configuration: {str(e)}")

    def _parse_int_env(self, var_name: str, default: int) -> int:
        """
        Parse an integer environment variable with validation.

        Args:
            var_name: Name of environment variable
            default: Default value if variable is not set

        Returns:
            Parsed integer value

        Raises:
            ConfigurationError: If variable is set but not a valid integer
        """
        value = os.getenv(var_name)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            raise ConfigurationError(
                f"{var_name} must be a valid integer, got: {value}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dictionary containing current configuration

        Raises:
            ConfigurationError: If configuration is not loaded
        """
        if self.config is None:
            raise ConfigurationError("Configuration not loaded")

        return self.config.copy()

    def get_provider(self) -> str:
        if self.config is None:
            # Default fallback if somehow not loaded, though init handles it
            return 'openrouter'
        return self.config.get('provider', 'openrouter')

    def get_base_url(self) -> str:
        if self.config is None:
            return 'https://openrouter.ai/api/v1'
        return self.config.get('base_url', 'https://openrouter.ai/api/v1')

    def get_api_key(self) -> str:
        """
        Get the API key.

        Returns:
            API key string

        Raises:
            ConfigurationError: If API key is not available
        """
        if not self.config or 'api_key' not in self.config:
            raise ConfigurationError("API key not available")

        return self.config['api_key']

    def get_model(self) -> str:
        """
        Get the default LLM model identifier.

        Returns:
            LLM model identifier string
        """
        return self.config['model'] if self.config else 'mistralai/mistral-7b-instruct'

    def get_timeout(self) -> int:
        """
        Get the API timeout setting.

        Returns:
            API timeout in seconds
        """
        return self.config['timeout'] if self.config else 30

    def get_max_retries(self) -> int:
        """
        Get the maximum retry attempts setting.

        Returns:
            Maximum retry attempts
        """
        return self.config['max_retries'] if self.config else 3

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key format.

        Args:
            api_key: API key string to validate

        Returns:
            True if API key appears valid, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            return False

        # Basic validation - reasonable length and format
        return len(api_key.strip()) >= 20 and 'sk-' in api_key.lower()


# Data Structures from Architecture
class ClusterMetadata(BaseModel):
    """Input data structure for cluster metadata"""
    cluster_id: str
    image_ids: list[str]
    text_metadata: list[str]  # Combined text from tags, titles, etc.
    cluster_size: int
    spatial_info: Optional[dict] = None  # Optional spatial characteristics

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "cluster_id": "cluster_001",
            "image_ids": ["img_001", "img_002", "img_003"],
            "text_metadata": [
                "paris eiffel tower landmark",
                "eiffel tower night lights",
                "paris cityscape with eiffel"
            ],
            "cluster_size": 3,
            "spatial_info": {
                "centroid": [48.8584, 2.2945],
                "bounding_box": [[48.85, 2.29], [48.86, 2.30]]
            }
        }
    })


class LabelResult(BaseModel):
    """Output data structure for label results"""
    label: str
    processing_info: dict
    timing: dict

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "label": "Eiffel Tower Paris Landmark",
            "processing_info": {
                "model_used": "mistralai/mistral-7b-instruct",
                "prompt_tokens": 45,
                "completion_tokens": 12,
                "total_tokens": 57
            },
            "timing": {
                "total_processing_time": 1.23,
                "llm_response_time": 0.87
            }
        }
    })


def handle_error(error: Exception, context: Optional[dict] = None) -> dict:
    """
    Centralized error handling function.

    Args:
        error: Exception to handle
        context: Additional context for error handling

    Returns:
        Dictionary containing error information and recovery suggestions
    """
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'context': context or {}
    }

    # Add specific handling for different error types
    if isinstance(error, ConfigurationError):
        error_info['recovery_suggestion'] = "Check .env file and API key configuration"
        error_info['severity'] = "HIGH"

    elif isinstance(error, InputValidationError):
        error_info['recovery_suggestion'] = "Validate input metadata structure and content"
        error_info['severity'] = "MEDIUM"

    elif isinstance(error, APICommunicationError):
        error_info['recovery_suggestion'] = "Check network connection and retry with exponential backoff"
        error_info['severity'] = "HIGH"

    elif isinstance(error, LLMProcessingError):
        error_info['recovery_suggestion'] = "Retry with different model or prompt formulation"
        error_info['severity'] = "MEDIUM"

    else:
        error_info['recovery_suggestion'] = "Unexpected error, check system logs"
        error_info['severity'] = "CRITICAL"

    return error_info


def validate_input_metadata(metadata: dict) -> bool:
    """
    Validate cluster metadata structure and content.

    Args:
        metadata: Input metadata dictionary

    Returns:
        True if metadata is valid, False otherwise

    Raises:
        InputValidationError: If metadata structure is invalid
    """
    try:
        # Validate using Pydantic model
        ClusterMetadata(**metadata)
        return True
    except ValidationError as e:
        error_details = {
            'validation_errors': str(e),
            'metadata_keys': list(metadata.keys()) if metadata else []
        }
        logger.error(f"Input validation failed: {error_details}")
        raise InputValidationError(f"Invalid metadata structure: {str(e)}")


def initialize_llm_client(
        api_key: str,
        model: Optional[str] = None,
        timeout: int = 30,
        base_url: str = 'https://openrouter.ai/api/v1'):
    """
    Initialize the LLM client configuration.

    Args:
        api_key: API key
        model: LLM model identifier (e.g., 'mistralai/mistral-7b-instruct')
        timeout: API request timeout in seconds
        base_url: Base URL for the API

    Returns:
        Dictionary containing client configuration

    Raises:
        ValueError: If api_key is invalid or empty (unless using Ollama/local)
    """
    # Validate API key (skip if using local/ollama which might have dummy key)
    is_local = 'localhost' in base_url or '127.0.0.1' in base_url or api_key == 'ollama'

    if not is_local:
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        # Basic API key validation
        if len(api_key.strip()) < 20:
            raise ValueError("API key appears invalid")

    # Validate model
    if model and len(model.split('/')) != 2 and not is_local:
        # Standardize warning but allow pass for local models which might be
        # just "llama3"
        pass

    # Return client configuration
    return {
        'api_key': api_key,
        'model': model or 'mistralai/mistral-7b-instruct',
        'timeout': timeout,
        'base_url': base_url
    }


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients."""

    @abc.abstractmethod
    def call_llm_api(self, prompt: str, max_tokens: int = 100,
                     temperature: float = 0.7) -> dict:
        pass


class UnifiedOpenAIClient(BaseLLMClient):
    """
    Generic client for any OpenAI-compatible API (OpenRouter, Ollama, vLLM, etc).
    Handles API communication, request formatting, response parsing,
    and error handling.
    """

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            extra_headers: Optional[dict] = None,
            timeout: int = 30,
            max_retries: int = 3):
        """
        Initialize Unified OpenAI client.

        Args:
            api_key: API key
            base_url: Base URL for the API
            model: LLM model identifier
            extra_headers: Optional extra headers for the request
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Merge default headers with any provider-specific headers
        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=headers
        )

        logger.info(
            f"Initialized Unified Client for {base_url} with model: {model}")

        # Check specific model requirements for OpenRouter
        if 'openrouter' in self.base_url:
            self._check_openrouter_limits()

    def _check_openrouter_limits(self):
        """
        Check rate limits based on model type for OpenRouter.
        """
        try:
            # Check for free specific model suffix
            if self.model.endswith(':free'):
                logger.info(
                    f"Detected free model '{self.model}'. Enforcing strict rate limit (20 RPM).")
                _global_rate_limiter.update_limit(20)
        except Exception as e:
            logger.error(f"Error checking initial rate limits: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((OpenAIError,
                                      APICommunicationError,
                                      Exception))
    )
    def call_llm_api(self, prompt: str, max_tokens: int = 100,
                     temperature: float = 0.7) -> dict:
        """
        Call LLM API with retry mechanism.

        Args:
            prompt: Input prompt for LLM
            max_tokens: Maximum tokens for response
            temperature: Creativity temperature (0-1)

        Returns:
            Dictionary containing LLM response and metadata

        Raises:
            APICommunicationError: If API communication fails
            LLMProcessingError: If LLM returns invalid response
        """
        import time
        start_time = time.time()

        try:
            # Enforce global rate limit before making request
            _global_rate_limiter.wait()

            logger.debug(
                f"Sending request to {self.base_url} with model {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout
            )

            # Extract content
            if not response.choices:
                raise LLMProcessingError("No choices returned from LLM API")

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMProcessingError("Empty content in LLM response")

            # Calculate timing and token usage
            llm_response_time = time.time() - start_time

            # OpenAI object access
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            return {
                'content': content.strip(),
                'model': response.model,
                'timing': {
                    'llm_response_time': llm_response_time,
                    'total_processing_time': llm_response_time
                },
                'tokens': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                },
                'raw_response': response.model_dump()
            }

        except OpenAIError as e:
            # Handle rate limits specifically if possible
            if "429" in str(e):
                logger.warning(
                    "Received 429 via OpenAI Client. Reducing rate limit.")
                current_limit = _global_rate_limiter.limit
                new_limit = max(5, int(current_limit * 0.5))
                _global_rate_limiter.update_limit(new_limit)

            raise APICommunicationError(
                f"OpenAI API communication failed: {str(e)}")
        except Exception as e:
            if isinstance(e, LLMProcessingError):
                raise
            raise LLMProcessingError(f"LLM processing failed: {str(e)}")


class LLMLabelingService:
    """
    Main service class for LLM-based cluster labeling.

    Orchestrates the entire labeling workflow including input validation,
    prompt engineering, LLM API calls, and response processing.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize LLMLabelingService with configuration.

        Args:
            config_manager: ConfigManager instance with loaded configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()

        provider = config_manager.get_provider()

        # Configure client based on provider
        if provider == 'ollama':
            api_key = self.config['api_key']
            base_url = config_manager.get_base_url()  # Already processed in ConfigManager
            extra_headers = None
        else:
            # Default to OpenRouter
            api_key = self.config['api_key']
            base_url = config_manager.get_base_url()
            extra_headers = {
                "HTTP-Referer": "https://your-app-name.com",
                "X-Title": "UnPointMaps LLM Labeling System"
            }

        self.api_client = UnifiedOpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=self.config['model'],
            extra_headers=extra_headers,
            timeout=self.config['timeout'],
            max_retries=self.config['max_retries']
        )

        logger.info(
            f"LLMLabelingService initialized successfully (Provider: {provider})")

        # Initialize Database Manager for Caching
        self.db_manager = DatabaseManager()

        # Initialize Embedding Service for Representative Selection
        self.embedding_service = EmbeddingService.get_instance()
        # Ensure model is checked/loaded, though server.py likely did it.
        # It's a singleton, so load_model() is idempotent if logic is correct
        # (I checked it is safe)
        if hasattr(
                self.embedding_service,
                'model') and self.embedding_service.model is None:
            self.embedding_service.load_model()

    def validate_and_prepare_input(
            self, cluster_metadata: dict) -> ClusterMetadata:
        """
        Validate and prepare cluster metadata for processing.

        Args:
            cluster_metadata: Raw cluster metadata dictionary

        Returns:
            Validated ClusterMetadata object

        Raises:
            InputValidationError: If input validation fails
        """
        try:
            # Validate using Pydantic model
            validated_metadata = ClusterMetadata(**cluster_metadata)

            # Additional validation: ensure text metadata is not empty
            if not validated_metadata.text_metadata or len(
                    validated_metadata.text_metadata) == 0:
                raise InputValidationError("text_metadata cannot be empty")

            # Ensure all text metadata items are strings
            for i, text in enumerate(validated_metadata.text_metadata):
                if not isinstance(text, str):
                    raise InputValidationError(
                        f"text_metadata item at index {i} is not a string")

                if not text.strip():
                    raise InputValidationError(
                        f"text_metadata item at index {i} is empty")

            return validated_metadata

        except ValidationError as e:
            error_details = {
                'validation_errors': str(e), 'metadata_keys': list(
                    cluster_metadata.keys()) if cluster_metadata else []}
            logger.error(f"Input validation failed: {error_details}")
            raise InputValidationError(f"Invalid metadata structure: {str(e)}")

    def _select_representative_sentences(
            self,
            texts: List[str],
            top_k: int = 20) -> List[str]:
        """
        Select representative sentences using centroid-based ranking.

        Args:
            texts: List of input text strings
            top_k: Number of sentences to select

        Returns:
            List of selected representative sentences
        """
        if not self.embedding_service or not texts:
            return texts[:top_k]

        try:
            top_indices = self.embedding_service.select_representative_indices(
                texts, top_k=top_k)
            selected_texts = [texts[i] for i in top_indices]
            return selected_texts
        except Exception as e:
            logger.warning(
                f"Error in representative sentence selection: {e}. Falling back to simple selection.")
            return texts[:top_k]

    def construct_labeling_prompt(
            self,
            metadata: ClusterMetadata,
            task_description: Optional[str] = None) -> str:
        """
        Construct optimized prompt for cluster labeling.

        Args:
            metadata: Validated cluster metadata
            task_description: Optional task-specific description

        Returns:
            Formatted prompt string for LLM
        """
        # Default task description if not provided
        if task_description is None:
            task_description = (
                "You are an expert in image analysis and geographic labeling. "
                "Summarize the provided representative sentences to generate a concise label.")

        # Select representative sentences using centroid strategy
        selected_texts = self._select_representative_sentences(
            metadata.text_metadata, top_k=20)
        combined_text = '\n'.join(selected_texts)

        # Construct prompt with clear instructions
        prompt = f"""{task_description}
Representative Sentences (selected by centroid proximity):
{combined_text}

Instructions:
1. These sentences are the most representative of the cluster's content.
2. Summarize these sentences into ONE concise sentence capturing the average meaning.
3. Length constraint: Keep it under 15 words.
4. IMPORTANT: Output ONLY the summary sentence. Do not include quotes or explanations.

Your Summary:"""

        return prompt

    def _generate_cache_key(
            self,
            prompt: str,
            model: str,
            temperature: float) -> str:
        """
        Generate a unique cache key for the LLM request.
        """
        # Combine prompt and parameters
        data = f"{prompt}_{model}_{temperature}"
        return hashlib.md5(data.encode('utf-8')).hexdigest()

    def generate_cluster_label(self, cluster_metadata: dict,
                               max_tokens: int = 100,
                               temperature: float = 0.7) -> LabelResult:
        """
        Generate a descriptive label for a cluster using LLM.

        Args:
            cluster_metadata: Dictionary containing cluster text metadata
            max_tokens: Maximum tokens for LLM response
            temperature: Creativity temperature (0-1)

        Returns:
            LabelResult object containing generated label and metadata

        Raises:
            LLMProcessingError: If label generation fails
            InputValidationError: If input metadata is invalid
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Validate and prepare input
            logger.info(
                f"Validating input for cluster {cluster_metadata.get('cluster_id', 'unknown')}")
            validated_metadata = self.validate_and_prepare_input(
                cluster_metadata)

            # Step 2: Construct optimized prompt
            logger.info(
                f"Constructing LLM prompt for cluster {validated_metadata.cluster_id}")
            prompt = self.construct_labeling_prompt(validated_metadata)

            # Log first 200 chars
            logger.info(f"Generated prompt: {prompt[:200]}...")

            # --- Caching Check ---
            cache_key = self._generate_cache_key(
                prompt, self.config['model'], temperature)
            cached_data = self.db_manager.get_cached_llm_label(cache_key)
            if cached_data:
                logger.info(
                    f"Using cached label for cluster {validated_metadata.cluster_id}")
                return LabelResult(**cached_data)

            # Step 3: Call LLM API
            logger.info(
                f"Calling LLM API for cluster {validated_metadata.cluster_id}")
            llm_response = self.api_client.call_llm_api(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Step 4: Process raw LLM response
            logger.info(
                f"Processing LLM response for cluster {validated_metadata.cluster_id}")
            processed_label = self._process_llm_response(
                llm_response['content'],
                validated_metadata
            )

            logger.info(
                f"Generated label '{processed_label}' for cluster {validated_metadata.cluster_id}")

            # Calculate total processing time
            total_processing_time = time.time() - start_time

            # Create result object data
            result_data: Dict[str, Any] = {
                'label': processed_label,
                'processing_info': {
                    'model_used': llm_response['model'],
                    'prompt_tokens': llm_response['tokens']['prompt_tokens'],
                    'completion_tokens': llm_response['tokens']['completion_tokens'],
                    'total_tokens': llm_response['tokens']['total_tokens'],
                    'processing_steps': [
                        'input_validation',
                        'prompt_construction',
                        'llm_api_call',
                        'response_processing',
                        'label_validation'
                    ]
                },
                'timing': {
                    'total_processing_time': total_processing_time,
                    'llm_response_time': llm_response.get('timing', {}).get('llm_response_time', 0),
                    'pre_processing_time': 0
                }
            }

            # Cache the result
            self.db_manager.save_cached_llm_label(cache_key, result_data)

            # Create result object
            result = LabelResult(
                label=result_data['label'],
                processing_info=result_data['processing_info'],
                timing=result_data['timing']
            )

            logger.info(f"Successfully generated label: {processed_label}")
            return result

        except Exception as e:
            error_info = handle_error(e, {
                'cluster_id': cluster_metadata.get('cluster_id', 'unknown'),
                'workflow': 'generate_cluster_label'
            })
            logger.error(f"Label generation failed: {error_info}")
            raise LLMProcessingError(f"Label generation failed: {str(e)}")

    def process_batch(self,
                      cluster_metadata_list: List[dict],
                      max_workers: int = 1) -> Dict[int,
                                                    str]:
        """
        Process a batch of clusters in parallel.

        Args:
            cluster_metadata_list: List of cluster metadata dictionaries.
                                 Each must have 'cluster_id' like 'cluster_123'.
            max_workers: Number of parallel workers

        Returns:
            Dictionary mapping cluster index (int) to label
        """
        results = {}

        def _process_single(meta):
            # Extract index from cluster_id "cluster_123" -> 123
            c_id = meta.get('cluster_id', 'cluster_0')
            try:
                idx = int(c_id.split('_')[1])
            except BaseException:
                idx = -1

            try:
                result = self.generate_cluster_label(meta)
                return idx, result.label, None
            except Exception as e:
                return idx, c_id, str(e)

        logger.info(
            f"Starting batch processing with {max_workers} workers for {len(cluster_metadata_list)} clusters")
        print(
            f"\nProcessing {len(cluster_metadata_list)} clusters with {max_workers} parallel workers...")

        # Display progress bar or counter
        completed_count = 0
        total_count = len(cluster_metadata_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {executor.submit(
                _process_single, meta): meta for meta in cluster_metadata_list}

            for future in as_completed(future_to_meta):
                idx, label, error = future.result()
                results[idx] = label
                completed_count += 1

                if error:
                    logger.error(f"Error processing cluster {idx}: {error}")
                    print(
                        f"[{completed_count}/{total_count}] Cluster {idx} failed: {error}")
                else:
                    print(
                        f"[{completed_count}/{total_count}] Cluster {idx} labelled: {label}")

        return results

    def _process_llm_response(self, raw_label: str,
                              metadata: ClusterMetadata) -> str:
        """
        Process and clean raw LLM response.

        Args:
            raw_label: Raw label string from LLM
            metadata: Original cluster metadata

        Returns:
            Cleaned and processed label string

        Raises:
            PostProcessingError: If response processing fails
        """
        try:
            # Basic cleaning
            processed = raw_label.strip()

            # Remove any leading/trailing quotes or markers
            if processed.startswith('"') and processed.endswith('"'):
                processed = processed[1:-1]
            elif processed.startswith("'") and processed.endswith("'"):
                processed = processed[1:-1]

            # Remove any trailing punctuation that might be excessive
            while processed and processed[-1] in '.,;:!?' and len(
                    processed) > 1:
                processed = processed[:-1]

            # Capitalize first letter
            if processed:
                processed = processed[0].upper(
                ) + processed[1:] if processed else processed

            # Basic validation
            if not processed:
                raise PostProcessingError("Processed label is empty")

            if len(processed) < 3:
                raise PostProcessingError("Processed label is too short")

            if len(processed) > 100:
                raise PostProcessingError("Processed label is too long")

            return processed

        except Exception as e:
            raise PostProcessingError(f"Response processing failed: {str(e)}")


def main_llm_labelling_workflow(
        cluster_metadata: dict,
        env_path: str = '.env') -> dict:
    """
    Main workflow for LLM-based cluster labeling.

    Args:
        cluster_metadata: Dictionary containing cluster text metadata
        env_path: Path to .env file

    Returns:
        Dictionary containing:
        - 'label': Final generated label
        - 'processing_info': Detailed processing information
        - 'timing': Performance metrics

    Raises:
        LLMProcessingError: If any step in workflow fails
    """
    try:
        # Initialize configuration
        config_manager = ConfigManager(env_path)

        # Initialize labeling service
        labeling_service = LLMLabelingService(config_manager)

        # Generate cluster label
        result = labeling_service.generate_cluster_label(cluster_metadata)

        # Convert LabelResult to dictionary for compatibility
        return result.model_dump()
    except Exception as e:
        error_info = handle_error(
            e, {'workflow': 'main_llm_labelling_workflow'})
        logger.error(f"LLM workflow failed: {error_info}")
        raise LLMProcessingError(f"LLM workflow failed: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Example cluster metadata
    example_metadata = {
        "cluster_id": "test_cluster_001",
        "image_ids": ["test_img_001", "test_img_002"],
        "text_metadata": [
            "test image with landmarks",
            "cityscape photography"
        ],
        "cluster_size": 2
    }

    try:
        # Test configuration loading
        print("Testing ConfigManager...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print(f"Configuration loaded: {config}")

        # Test main workflow
        print("\nTesting main workflow...")
        result = main_llm_labelling_workflow(example_metadata)
        print(f"Workflow result: {result}")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("This is expected if .env file contains placeholder API key")
