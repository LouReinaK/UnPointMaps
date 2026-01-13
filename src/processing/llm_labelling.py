# LLM Labeling System - Environment Configuration and Core Functionality
# Based on the architecture specified in plans/llm_labelling_architecture.md

import os
import logging
import time
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
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
            load_dotenv(self.env_path)
            
            # Validate required API key
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key or not api_key.strip():
                raise ConfigurationError("OPENROUTER_API_KEY not set or empty")
            
            # Basic API key validation - reasonable minimum length
            if len(api_key.strip()) < 20:
                raise ConfigurationError("OPENROUTER_API_KEY appears invalid - too short")
            
            # Load and validate other configuration parameters
            config = {
                'api_key': api_key.strip(),
                'model': os.getenv('DEFAULT_LLM_MODEL', 'mistralai/mistral-7b-instruct'),
                'timeout': self._parse_int_env('API_TIMEOUT', 30),
                'max_retries': self._parse_int_env('MAX_RETRIES', 3),
                'env_path': self.env_path
            }
            
            # Validate model name format
            if not config['model'] or len(config['model'].split('/')) != 2:
                raise ConfigurationError("DEFAULT_LLM_MODEL must be in format 'provider/model-name'")
            
            # Validate timeout and retry values
            if config['timeout'] <= 0:
                raise ConfigurationError("API_TIMEOUT must be a positive integer")
            
            if config['max_retries'] < 0:
                raise ConfigurationError("MAX_RETRIES must be a non-negative integer")
            
            self.config = config
            logger.info(f"Configuration loaded successfully from {self.env_path}")
            
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
    
    def get_api_key(self) -> str:
        """
        Get the OpenRouter API key.
        
        Returns:
            OpenRouter API key string
            
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
        schema_extra = {
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
    confidence: float
    processing_info: dict
    timing: dict

    class Config:
        schema_extra = {
            "example": {
                "label": "Eiffel Tower Paris Landmark",
                "confidence": 0.95,
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


def initialize_llm_client(api_key: str, model: Optional[str] = None, timeout: int = 30):
    """
    Initialize the LLM client with OpenRouter API configuration.
    
    Args:
        api_key: OpenRouter API key
        model: LLM model identifier (e.g., 'mistralai/mistral-7b-instruct')
        timeout: API request timeout in seconds
        
    Returns:
        Dictionary containing client configuration (placeholder for actual client)
        
    Raises:
        ValueError: If api_key is invalid or empty
        ConnectionError: If unable to establish connection
    """
    # Validate API key
    if not api_key or not api_key.strip():
        raise ValueError("API key cannot be empty")
    
    # Basic API key validation
    if len(api_key.strip()) < 20:
        raise ValueError("API key appears invalid")
    
    # Validate model
    if model and len(model.split('/')) != 2:
        raise ValueError("Model must be in format 'provider/model-name'")
    
    # Return client configuration (actual client implementation would go here)
    return {
        'api_key': api_key,
        'model': model or 'mistralai/mistral-7b-instruct',
        'timeout': timeout,
        'base_url': 'https://openrouter.ai/api/v1'
    }


class OpenRouterAPIClient:
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
        
        # Validate configuration
        if not self._validate_api_key(api_key):
            raise ConfigurationError("Invalid OpenRouter API key")
            
        if not self._validate_model(model):
            raise ConfigurationError("Invalid model identifier")
        
        # Import requests library
        import requests
        self.requests = requests
        
        logger.info(f"OpenRouterAPIClient initialized with model: {model}")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if API key appears valid, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Basic validation - reasonable length and format
        stripped_key = api_key.strip()
        return len(stripped_key) >= 20 and 'sk-' in stripped_key.lower()
    
    def _validate_model(self, model: str) -> bool:
        """
        Validate model identifier format.
        
        Args:
            model: Model identifier to validate
            
        Returns:
            True if model identifier is valid, False otherwise
        """
        if not model or not isinstance(model, str):
            return False
        
        # Model should be in format 'provider/model-name'
        parts = model.split('/')
        return len(parts) == 2 and all(part.strip() for part in parts)
    
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
        
        # Initialize API client
        self.api_client = OpenRouterAPIClient(
            api_key=self.config['api_key'],
            model=self.config['model'],
            timeout=self.config['timeout'],
            max_retries=self.config['max_retries']
        )
        
        logger.info("LLMLabelingService initialized successfully")
    
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

Please provide a concise, descriptive label for this cluster. The label should:
1. Be informative and specific
2. Capture the main theme or subject
3. Be suitable for use in data analysis and visualization
4. Be 3-8 words in length

Label:"""
        
        return prompt
    
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
            
            # Step 5: Validate and refine label
            final_label, confidence = self._validate_and_refine_label(
                processed_label,
                validated_metadata
            )
            
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            
            # Create result object
            result = LabelResult(
                label=final_label,
                confidence=confidence,
                processing_info={
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
                timing={
                    'total_processing_time': total_processing_time,
                    'llm_response_time': llm_response['timing']['llm_response_time'],
                    'pre_processing_time': llm_response['timing']['llm_response_time'] - total_processing_time
                }
            )
            
            logger.info(f"Successfully generated label: {final_label}")
            return result
            
        except Exception as e:
            error_info = handle_error(e, {
                'cluster_id': cluster_metadata.get('cluster_id', 'unknown'),
                'workflow': 'generate_cluster_label'
            })
            logger.error(f"Label generation failed: {error_info}")
            raise LLMProcessingError(f"Label generation failed: {str(e)}")
    
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
    
    def _validate_and_refine_label(self, label: str,
                                   metadata: ClusterMetadata) -> tuple:
        """
        Validate and refine the generated label.
        
        Args:
            label: Processed label string
            metadata: Original cluster metadata
            
        Returns:
            Tuple of (final_label, confidence_score)
            
        Raises:
            PostProcessingError: If label validation fails
        """
        try:
            # Basic validation
            if not label or not isinstance(label, str):
                raise PostProcessingError("Label is invalid or empty")
            
            # Calculate confidence based on label quality heuristics
            confidence = self._calculate_label_confidence(label, metadata)
            
            # Apply refinements if needed
            refined_label = self._apply_label_refinements(label, metadata)
            
            return refined_label, confidence
            
        except Exception as e:
            raise PostProcessingError(f"Label validation failed: {str(e)}")
    
    def _calculate_label_confidence(self, label: str,
                                   metadata: ClusterMetadata) -> float:
        """
        Calculate confidence score for generated label.
        
        Args:
            label: Generated label string
            metadata: Original cluster metadata
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence based on label characteristics
        confidence = 0.7  # Base confidence
        
        # Adjust based on label length (optimal length gets higher confidence)
        label_length = len(label.split())
        if 3 <= label_length <= 8:
            confidence += 0.1
        elif label_length < 3 or label_length > 12:
            confidence -= 0.1
        
        # Check if label contains relevant keywords from metadata
        metadata_text = ' '.join(metadata.text_metadata).lower()
        label_lower = label.lower()
        
        # Extract keywords from metadata
        metadata_keywords = set()
        for text in metadata.text_metadata:
            words = text.lower().split()
            # Filter out common words
            common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or'}
            metadata_keywords.update(word for word in words if word not in common_words and len(word) > 2)
        
        # Check if any metadata keywords appear in label
        label_words = set(label_lower.split())
        keyword_matches = metadata_keywords.intersection(label_words)
        
        if keyword_matches:
            # Higher confidence if label contains relevant keywords
            keyword_coverage = len(keyword_matches) / len(metadata_keywords) if metadata_keywords else 0
            confidence += min(keyword_coverage * 0.2, 0.15)
        
        # Ensure confidence is within bounds
        return max(0.1, min(1.0, confidence))
    
    def _apply_label_refinements(self, label: str,
                                metadata: ClusterMetadata) -> str:
        """
        Apply refinements to improve label quality.
        
        Args:
            label: Generated label string
            metadata: Original cluster metadata
            
        Returns:
            Refined label string
        """
        # For now, return label as-is
        # Future enhancements could include:
        # - Synonym expansion
        # - Grammar correction
        # - Domain-specific refinements
        return label


def main_llm_labelling_workflow(cluster_metadata: dict, env_path: str = '.env') -> dict:
    """
    Main workflow for LLM-based cluster labeling.
    
    Args:
        cluster_metadata: Dictionary containing cluster text metadata
        env_path: Path to .env file
        
    Returns:
        Dictionary containing:
        - 'label': Final generated label
        - 'confidence': Confidence score
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