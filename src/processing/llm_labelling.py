# LLM Labeling System - Environment Configuration and Core Functionality
# Based on the architecture specified in plans/llm_labelling_architecture.md

import os
import logging
import time
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import abc
import hashlib  # Added for caching
import json     # Added for caching

from src.database.manager import DatabaseManager # Added for caching

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
                 max_per_minute = 1 # Avoid division by zero
            self.limit = max_per_minute
            self.delay = 60.0 / max_per_minute
            logger.info(f"Rate limit updated to {max_per_minute} requests/minute")

    def wait(self):
        """Blocks until a request can be made."""
        with self.lock:
            now = time.time()
            # Calculate when the next request is allowed
            next_allowed_time = self.last_call + self.delay
            wait_time = next_allowed_time - now
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_call = time.time() # Update to now (after sleep)

# Initialize global rate limiter instance
_global_rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)


# Configure logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

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
        self.config = None
        self._validate_env_file()
        self._load_environment()
    
    def _validate_env_file(self) -> None:
        """
        Validate that the environment file exists and is readable.
        
        Raises:
            ConfigurationError: If .env file is missing or not readable
        """
        if not os.path.exists(self.env_path):
            raise ConfigurationError(f"Environment file not found: {self.env_path}")
        
        if not os.access(self.env_path, os.R_OK):
            raise ConfigurationError(f"Cannot read environment file: {self.env_path}")
    
    def _load_environment(self) -> None:
        """
        Load environment variables from .env file and validate required settings.
        
        Raises:
            ConfigurationError: If required environment variables are missing or invalid
        """
        try:
            # Load environment variables from .env file
            # Use override=True to ensure .env changes are picked up even if env vars exist
            load_dotenv(self.env_path, override=True)
            
            # Identify provider
            provider = os.getenv('LLM_PROVIDER', 'openrouter').lower()
            
            config = {
                'provider': provider,
                'model': os.getenv('DEFAULT_LLM_MODEL', 'mistralai/mistral-7b-instruct'),
                'timeout': self._parse_int_env('API_TIMEOUT', 30),
                'max_retries': self._parse_int_env('MAX_RETRIES', 3),
                'env_path': self.env_path
            }

            if provider == 'ollama':
                ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                config['ollama_base_url'] = ollama_url
                config['api_key'] = 'ollama'
                logger.info(f"Using provider: {provider} with base URL: {ollama_url}")
            else:
                # OpenRouter Specifics
                api_key = os.getenv('OPENROUTER_API_KEY')
                if not api_key or not api_key.strip():
                     raise ConfigurationError("OPENROUTER_API_KEY not set or empty")
                if len(api_key.strip()) < 20:
                     raise ConfigurationError("OPENROUTER_API_KEY appears invalid - too short")
                config['api_key'] = api_key.strip()

            self.config = config
            logger.info(f"Configuration loaded successfully from {self.env_path} (Provider: {provider})")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load environment configuration: {str(e)}")
    
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
            raise ConfigurationError(f"{var_name} must be a valid integer, got: {value}")
    
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

    def get_ollama_base_url(self) -> str:
        if self.config is None:
             return 'http://localhost:11434'
        return self.config.get('ollama_base_url', 'http://localhost:11434')

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

    class Config:
        json_schema_extra = {
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
        }


class LabelResult(BaseModel):
    """Output data structure for label results"""
    label: str
    processing_info: dict
    timing: dict

    class Config:
        json_schema_extra = {
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
        }


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


def initialize_llm_client(api_key: str, model: Optional[str] = None, timeout: int = 30, 
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
        # Standardize warning but allow pass for local models which might be just "llama3"
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


class OpenRouterAPIClient(BaseLLMClient):
    """
    Client for interacting with OpenRouter API for LLM services.
    
    Handles API communication, request formatting, response parsing,
    and error handling for OpenRouter LLM API calls.
    """
    
    def __init__(self, api_key: str, model: str = 'mistralai/mistral-7b-instruct',
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize OpenRouter API client.
        
        Args:
            api_key: OpenRouter API key
            model: LLM model identifier (provider/model-name)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            
        Raises:
            ConfigurationError: If API key or model is invalid
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = 'https://openrouter.ai/api/v1'
        
        # Import requests library
        import requests
        self.requests = requests
        
        logger.info(f"OpenRouterAPIClient initialized with model: {model}")
        
        # Check rate limits and model type on initialization
        self._check_initial_limits()
        
    def _check_initial_limits(self):
        """
        Check rate limits based on model type and API key status.
        Dynamically limits request rate for free models or restricted keys.
        """
        try:
            # 1. Check for free specific model suffix
            if self.model.endswith(':free'):
                logger.info(f"Detected free model '{self.model}'. Enforcing strict rate limit (20 RPM).")
                _global_rate_limiter.update_limit(20)
                return

            # 2. Check API key status against OpenRouter /key endpoint
            try:
                headers = {'Authorization': f'Bearer {self.api_key}'}
                response = self.requests.get(f'{self.base_url}/key', headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json().get('data', {})
                    is_free_tier = data.get('is_free_tier', False)
                    # Note: rate_limit object is deprecated, so we rely on is_free_tier or other indicators
                    
                    if is_free_tier and self.model.endswith(':free'):
                         _global_rate_limiter.update_limit(20)
                         logger.info("Detected free tier user on free model. Limit set to 20 RPM.")
                    
                    # Log credit info for debugging
                    usage = data.get('usage', 'unknown')
                    limit = data.get('limit', 'unknown')
                    logger.info(f"API Key Stats - Usage: {usage}, Limit: {limit}, Free Tier: {is_free_tier}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch key info from OpenRouter: {e}")
                
        except Exception as e:
            logger.error(f"Error checking initial rate limits: {e}")

    def _construct_request_headers(self) -> dict:
        """
        Construct HTTP headers for OpenRouter API request.
        
        Returns:
            Dictionary containing request headers
        """
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://your-app-name.com',  # Required by OpenRouter
            'X-Title': 'UnPointMaps LLM Labeling System'
        }
    
    def _construct_request_payload(self, prompt: str, max_tokens: int = 100,
                                   temperature: float = 0.7) -> dict:
        """
        Construct request payload for OpenRouter API.
        
        Args:
            prompt: Input prompt for LLM
            max_tokens: Maximum tokens for response
            temperature: Creativity temperature (0-1)
            
        Returns:
            Dictionary containing request payload
        """
        return {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,
                                      APICommunicationError,
                                      Exception))
    )
    def call_llm_api(self, prompt: str, max_tokens: int = 100,
                    temperature: float = 0.7) -> dict:
        """
        Call OpenRouter API with retry mechanism.
        
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

            # Construct request
            headers = self._construct_request_headers()
            payload = self._construct_request_payload(prompt, max_tokens, temperature)
            
            logger.debug(f"Sending request to OpenRouter API: {payload}")
            
            # Make API request
            response = self.requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            # Special handling for Rate Limits (429)
            if response.status_code == 429:
                logger.warning("Received 429 Too Many Requests. Reducing rate limit dynamically.")
                # Reduce limit by half, minimum 5 RPM
                current_limit = _global_rate_limiter.limit
                new_limit = max(5, int(current_limit * 0.5))
                _global_rate_limiter.update_limit(new_limit)
                
                # Wait a bit longer before retry (managed by 'tenacity', but we can log)
                raise APICommunicationError("Rate limit exceeded (429)")

            # Check response status
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Validate response structure
            if not response_data or 'choices' not in response_data:
                raise LLMProcessingError("Invalid response structure from LLM API")
            
            if not response_data['choices']:
                raise LLMProcessingError("No choices returned from LLM API")
            
            # Extract and validate content
            content = response_data['choices'][0]['message']['content']
            if not content or not content.strip():
                raise LLMProcessingError("Empty content in LLM response")
            
            # Calculate timing and token usage
            llm_response_time = time.time() - start_time
            prompt_tokens = response_data.get('usage', {}).get('prompt_tokens', 0)
            completion_tokens = response_data.get('usage', {}).get('completion_tokens', 0)
            total_tokens = response_data.get('usage', {}).get('total_tokens', 0)
            
            return {
                'content': content.strip(),
                'model': response_data.get('model', self.model),
                'timing': {
                    'llm_response_time': llm_response_time,
                    'total_processing_time': llm_response_time
                },
                'tokens': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                },
                'raw_response': response_data
            }
            
        except requests.exceptions.RequestException as e:
            raise APICommunicationError(f"API communication failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise LLMProcessingError(f"Invalid response format from LLM API: {str(e)}")
        except Exception as e:
            # During retry, if we get connection/timeout errors or generic exceptions from requests, treat them as API communication errors
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                           requests.exceptions.RequestException, Exception)):
                raise APICommunicationError(f"API communication failed: {str(e)}")
            elif isinstance(e, LLMProcessingError):
                # Re-raise LLMProcessingError as-is
                raise
            else:
                raise LLMProcessingError(f"LLM processing failed: {str(e)}")


class OllamaClient(BaseLLMClient):
    """
    Client for interacting with local Ollama instance.
    """
    def __init__(self, base_url: str, model: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        import requests
        self.requests = requests
        logger.info(f"OllamaClient initialized with model: {model} at {base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def call_llm_api(self, prompt: str, max_tokens: int = 100,
                    temperature: float = 0.7) -> dict:
        import time
        start_time = time.time()
        
        # Ollama API format
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            # Enforce global rate limit (even for local, to avoid system freeze)
            # Maybe less strict but good practice
            # _global_rate_limiter.wait() 

            response = self.requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get('message', {}).get('content', '')
            
            # Extract metrics if available
            # Ollama returns durations in nanoseconds
            total_duration = data.get('total_duration', 0) / 1e9
            
            return {
                'content': content.strip(),
                'model': data.get('model', self.model),
                'timing': {
                    'llm_response_time': total_duration,
                    'total_processing_time': time.time() - start_time
                },
                'tokens': {
                    'prompt_tokens': data.get('prompt_eval_count', 0),
                    'completion_tokens': data.get('eval_count', 0),
                    'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
                },
                'raw_response': data
            }

        except Exception as e:
            raise APICommunicationError(f"Ollama API failed: {str(e)}")


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
        
        if provider == 'ollama':
            self.api_client = OllamaClient(
                base_url=config_manager.get_ollama_base_url(),
                model=self.config['model'],
                timeout=self.config['timeout']
            )
        else:
            # Default to OpenRouter
            self.api_client = OpenRouterAPIClient(
                api_key=self.config['api_key'],
                model=self.config['model'],
                timeout=self.config['timeout'],
                max_retries=self.config['max_retries']
            )
        
        logger.info(f"LLMLabelingService initialized successfully (Provider: {provider})")

        # Initialize Database Manager for Caching
        self.db_manager = DatabaseManager()
    
    def validate_and_prepare_input(self, cluster_metadata: dict) -> ClusterMetadata:
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
            if not validated_metadata.text_metadata or len(validated_metadata.text_metadata) == 0:
                raise InputValidationError("text_metadata cannot be empty")
            
            # Ensure all text metadata items are strings
            for i, text in enumerate(validated_metadata.text_metadata):
                if not isinstance(text, str):
                    raise InputValidationError(f"text_metadata item at index {i} is not a string")
                
                if not text.strip():
                    raise InputValidationError(f"text_metadata item at index {i} is empty")
            
            return validated_metadata
            
        except ValidationError as e:
            error_details = {
                'validation_errors': str(e),
                'metadata_keys': list(cluster_metadata.keys()) if cluster_metadata else []
            }
            logger.error(f"Input validation failed: {error_details}")
            raise InputValidationError(f"Invalid metadata structure: {str(e)}")
    
    def construct_labeling_prompt(self, metadata: ClusterMetadata,
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
                "Generate a concise, descriptive label for this cluster of images "
                "based on the provided text metadata. The label should be informative "
                "and capture the main theme or subject of the images."
            )
        
        # Combine text metadata into a single string
        combined_text = '\n'.join(metadata.text_metadata)
        
        # Construct prompt with clear instructions
        prompt = f"""{task_description}

Cluster Information:
- Cluster ID: {metadata.cluster_id}
- Number of images: {metadata.cluster_size}
- Image IDs: {', '.join(metadata.image_ids)}

Text Metadata (from images in this cluster):
{combined_text}

Instructions:
1. Analyze the metadata to identify the common theme, location, or event.
2. Create a specific, short label (2-6 words).
3. IMPORTANT: Output ONLY the label text. Do not include "Label:", quotes, explanations, or any other text.

Example Output:
Eiffel Tower Night View

Your Label:"""
        
        return prompt
        
        return prompt
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float) -> str:
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
            validated_metadata = self.validate_and_prepare_input(cluster_metadata)
            
            # Step 2: Construct optimized prompt
            prompt = self.construct_labeling_prompt(validated_metadata)
            
            logger.debug(f"Generated prompt: {prompt[:200]}...")  # Log first 200 chars

            # --- Caching Check ---
            cache_key = self._generate_cache_key(prompt, self.config['model'], temperature)
            cached_data = self.db_manager.get_cached_llm_label(cache_key)
            if cached_data:
                logger.info(f"Using cached label for cluster {validated_metadata.cluster_id}")
                return LabelResult(**cached_data)
            
            # Step 3: Call LLM API
            llm_response = self.api_client.call_llm_api(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Step 4: Process raw LLM response
            processed_label = self._process_llm_response(
                llm_response['content'],
                validated_metadata
            )
            
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            
            # Create result object data
            result_data = {
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
            result = LabelResult(**result_data)
            
            logger.info(f"Successfully generated label: {processed_label}")
            return result
            
        except Exception as e:
            error_info = handle_error(e, {
                'cluster_id': cluster_metadata.get('cluster_id', 'unknown'),
                'workflow': 'generate_cluster_label'
            })
            logger.error(f"Label generation failed: {error_info}")
            raise LLMProcessingError(f"Label generation failed: {str(e)}")
    
    def process_batch(self, cluster_metadata_list: List[dict], max_workers: int = 1) -> Dict[int, str]:
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
            except:
                idx = -1
            
            try:
                result = self.generate_cluster_label(meta)
                return idx, result.label, None
            except Exception as e:
                return idx, c_id, str(e)

        logger.info(f"Starting batch processing with {max_workers} workers for {len(cluster_metadata_list)} clusters")
        print(f"\nProcessing {len(cluster_metadata_list)} clusters with {max_workers} parallel workers...")
        
        # Display progress bar or counter
        completed_count = 0
        total_count = len(cluster_metadata_list)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {executor.submit(_process_single, meta): meta for meta in cluster_metadata_list}
            
            for future in as_completed(future_to_meta):
                idx, label, error = future.result()
                results[idx] = label
                completed_count += 1
                
                if error:
                    logger.error(f"Error processing cluster {idx}: {error}")
                    print(f"[{completed_count}/{total_count}] Cluster {idx} failed: {error}")
                else:
                    print(f"[{completed_count}/{total_count}] Cluster {idx} labelled: {label}")
                     
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
            while processed and processed[-1] in '.,;:!?' and len(processed) > 1:
                processed = processed[:-1]
            
            # Capitalize first letter
            if processed:
                processed = processed[0].upper() + processed[1:] if processed else processed
            
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

def main_llm_labelling_workflow(cluster_metadata: dict, env_path: str = '.env') -> dict:
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
        return result.dict()
        
    except Exception as e:
        error_info = handle_error(e, {'workflow': 'main_llm_labelling_workflow'})
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