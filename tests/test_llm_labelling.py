# Test suite for LLM Labeling System
# Tests all major components: configuration, API client, labeling
# workflow, and error handling

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.processing.llm_labelling import (
    ConfigManager,
    UnifiedOpenAIClient,
    LLMLabelingService,
    ClusterMetadata,
    LabelResult,
    main_llm_labelling_workflow,
    ConfigurationError,
    InputValidationError,
    LLMProcessingError,
    validate_input_metadata,
    handle_error
)


class TestConfigManager:
    """Test configuration management functionality"""

    def teardown_method(self):
        """Clean up environment variables after each test"""
        vars_to_clear = [
            'OPENROUTER_API_KEY',
            'DEFAULT_LLM_MODEL',
            'API_TIMEOUT',
            'MAX_RETRIES',
            'LLM_PROVIDER',
            'OLLAMA_BASE_URL',
            'OLLAMA_API_KEY',
            'OPENAI_BASE_URL',
            'OPENAI_API_KEY'
        ]
        for var in vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def test_valid_config_loading(self):
        """Test loading valid configuration from .env file"""
        # Create temporary .env file with valid configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            # Force provider to be openrouter
            f.write("LLM_PROVIDER=openrouter\n")
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            f.write("DEFAULT_LLM_MODEL=mistralai/mistral-7b-instruct\n")
            f.write("API_TIMEOUT=30\n")
            f.write("MAX_RETRIES=3\n")
            env_path = f.name

        try:
            # Test configuration loading
            config_manager = ConfigManager(env_path)
            config = config_manager.get_config()

            # Verify configuration values
            assert config['api_key'] == 'sk-valid_api_key_12345678901234567890'
            assert config['model'] == 'mistralai/mistral-7b-instruct'
            assert config['timeout'] == 30
            assert config['max_retries'] == 3
            assert config['env_path'] == env_path

        finally:
            os.unlink(env_path)

    def test_missing_env_file(self):
        """Test error handling for missing .env file"""
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager('nonexistent.env')

        assert "Environment file not found" in str(exc_info.value)

    def test_invalid_api_key(self):
        """Test error handling for invalid API key"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=\n")  # Empty key
            env_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigManager(env_path)

            assert "API key not found" in str(exc_info.value)
        finally:
            os.unlink(env_path)

    def test_missing_api_key(self):
        """Test error handling for missing API key"""
        # Ensure environment is clean
        if 'OPENROUTER_API_KEY' in os.environ:
            del os.environ['OPENROUTER_API_KEY']
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        if 'DEFAULT_LLM_MODEL' in os.environ:
            del os.environ['DEFAULT_LLM_MODEL']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("DEFAULT_LLM_MODEL=mistralai/mistral-7b-instruct\n")
            env_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigManager(env_path)

            assert "API key not found" in str(exc_info.value)
        finally:
            os.unlink(env_path)

    def test_invalid_model_format(self):
        """Test error handling for invalid model format"""
        # Ensure environment is clean
        if 'DEFAULT_LLM_MODEL' in os.environ:
            del os.environ['DEFAULT_LLM_MODEL']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            f.write("DEFAULT_LLM_MODEL=invalid_model\n")
            env_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigManager(env_path)

            assert "DEFAULT_LLM_MODEL must be in format" in str(exc_info.value)
        finally:
            os.unlink(env_path)

    def test_invalid_timeout(self):
        """Test error handling for invalid timeout value"""
        # Ensure environment is clean
        if 'API_TIMEOUT' in os.environ:
            del os.environ['API_TIMEOUT']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            f.write("API_TIMEOUT=-10\n")
            env_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigManager(env_path)

            assert "API_TIMEOUT must be a positive integer" in str(
                exc_info.value)
        finally:
            os.unlink(env_path)


class TestUnifiedOpenAIClient:
    """Test Unified OpenAI API client functionality"""

    def test_client_initialization(self):
        """Test successful client initialization"""
        api_key = "sk-valid_api_key_12345678901234567890"
        model = "mistralai/mistral-7b-instruct"
        base_url = 'https://openrouter.ai/api/v1'

        client = UnifiedOpenAIClient(api_key, base_url, model)

        assert client.api_key == api_key
        assert client.model == model
        assert client.timeout == 30
        assert client.max_retries == 3
        assert client.base_url == base_url

    @patch('src.processing.llm_labelling.OpenAI')
    def test_successful_api_call(self, mock_openai):
        """Test successful API call with valid response"""
        # Setup mock client and response
        mock_client_instance = MagicMock()
        mock_openai.return_value = mock_client_instance

        # Mock the chat completion create method
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = 'Test Label'
        mock_response.choices = [mock_choice]
        mock_response.model = 'mistralai/mistral-7b-instruct'

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 25
        mock_response.usage = mock_usage

        mock_response.model_dump.return_value = {'id': 'test_id'}

        mock_client_instance.chat.completions.create.return_value = mock_response

        # Create client and make API call
        client = UnifiedOpenAIClient(
            "sk-valid_api_key", "https://api.test", "model")
        result = client.call_llm_api("Test prompt")

        # Verify result structure
        assert result['content'] == 'Test Label'
        assert result['model'] == 'mistralai/mistral-7b-instruct'
        assert result['tokens']['prompt_tokens'] == 20
        assert result['tokens']['completion_tokens'] == 5
        assert result['tokens']['total_tokens'] == 25
        assert 'llm_response_time' in result['timing']

    @patch('src.processing.llm_labelling.OpenAI')
    @patch('src.processing.llm_labelling._global_rate_limiter')
    def test_api_communication_error(self, mock_limiter, mock_openai):
        """Test error handling for API communication failures"""
        mock_client_instance = MagicMock()
        mock_openai.return_value = mock_client_instance

        # Setup mock to raise exception
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "Connection failed")

        # Test API call with communication error
        client = UnifiedOpenAIClient(
            "sk-valid_api_key", "https://api.test", "model")

        # The retry mechanism will retry 3 times and then raise RetryError
        # containing APICommunicationError
        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            client.call_llm_api("Test prompt")

        # Check that the exception mechanism worked (might be RetryError wrapping APICommunicationError)
        # Verify the cause or string representation
        assert "LLMProcessingError" in str(
            exc_info.value) or "APICommunicationError" in str(
            exc_info.value) or "Connection failed" in str(
            exc_info.value)

    @patch('src.processing.llm_labelling.OpenAI')
    def test_invalid_api_response(self, mock_openai):
        """Test error handling for invalid API response format"""
        mock_client_instance = MagicMock()
        mock_openai.return_value = mock_client_instance

        # Setup mock response with invalid structure (no choices)
        mock_response = MagicMock()
        mock_response.choices = []

        mock_client_instance.chat.completions.create.return_value = mock_response

        # Test API call with invalid response
        client = UnifiedOpenAIClient(
            "sk-valid_api_key", "https://api.test", "model")

        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            client.call_llm_api("Test prompt")

        # Check the underlying exception
        cause = exc_info.value.last_attempt.exception()
        assert "No choices returned" in str(cause)


class TestInputValidation:
    """Test input validation functionality"""

    def test_valid_metadata(self):
        """Test validation of valid cluster metadata"""
        valid_metadata = {
            "cluster_id": "test_cluster",
            "image_ids": ["img1", "img2"],
            "text_metadata": ["test text 1", "test text 2"],
            "cluster_size": 2
        }

        result = validate_input_metadata(valid_metadata)
        assert result is True

    def test_invalid_metadata_structure(self):
        """Test error handling for invalid metadata structure"""
        invalid_metadata = {
            "cluster_id": "test_cluster",
            # Missing required fields
        }

        with pytest.raises(InputValidationError) as exc_info:
            validate_input_metadata(invalid_metadata)

        assert "Invalid metadata structure" in str(exc_info.value)

    def test_empty_text_metadata(self):
        """Test error handling for empty text metadata"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            invalid_metadata = {
                "cluster_id": "test_cluster",
                "image_ids": ["img1"],
                "text_metadata": [],
                "cluster_size": 1
            }

            config_manager = ConfigManager(env_path)
            service = LLMLabelingService(config_manager)

            with pytest.raises(InputValidationError) as exc_info:
                service.validate_and_prepare_input(invalid_metadata)

            assert "text_metadata cannot be empty" in str(exc_info.value)

        finally:
            os.unlink(env_path)


class TestLLMLabelingService:
    """Test the main labeling service functionality"""

    def teardown_method(self):
        """Clean up environment variables after each test"""
        vars_to_clear = [
            'OPENROUTER_API_KEY',
            'DEFAULT_LLM_MODEL',
            'API_TIMEOUT',
            'MAX_RETRIES',
            'LLM_PROVIDER',
            'OLLAMA_BASE_URL',
            'OLLAMA_API_KEY'
        ]
        for var in vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def test_service_initialization(self):
        """Test successful service initialization"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            config_manager = ConfigManager(env_path)
            service = LLMLabelingService(config_manager)

            assert service.config_manager == config_manager
            assert service.api_client is not None

        finally:
            os.unlink(env_path)

    def test_input_validation_and_preparation(self):
        """Test input validation and preparation"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            config_manager = ConfigManager(env_path)
            service = LLMLabelingService(config_manager)

            # Test valid input
            valid_metadata = {
                "cluster_id": "test_cluster",
                "image_ids": ["img1", "img2"],
                "text_metadata": ["test text 1", "test text 2"],
                "cluster_size": 2
            }

            result = service.validate_and_prepare_input(valid_metadata)
            assert isinstance(result, ClusterMetadata)
            assert result.cluster_id == "test_cluster"

            # Test invalid input
            invalid_metadata = {
                "cluster_id": "test_cluster",
                "image_ids": ["img1"],
                "text_metadata": [],
                "cluster_size": 1
            }

            with pytest.raises(InputValidationError) as exc_info:
                service.validate_and_prepare_input(invalid_metadata)

            assert "text_metadata cannot be empty" in str(exc_info.value)

        finally:
            os.unlink(env_path)

    def test_prompt_construction(self):
        """Test prompt construction functionality"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            f.write("DEFAULT_LLM_MODEL=mistralai/mistral-7b-instruct\n")
            env_path = f.name

        try:
            config_manager = ConfigManager(env_path)
            service = LLMLabelingService(config_manager)

            # Create test metadata
            metadata = ClusterMetadata(
                cluster_id="test_cluster",
                image_ids=["img1", "img2"],
                text_metadata=["paris eiffel tower", "eiffel tower night"],
                cluster_size=2
            )

            # Test prompt construction
            prompt = service.construct_labeling_prompt(metadata)

            # Verify prompt contains expected elements
            assert "paris eiffel tower" in prompt
            assert "eiffel tower night" in prompt
            assert "Your Summary:" in prompt

        finally:
            os.unlink(env_path)

    @patch('src.processing.llm_labelling.UnifiedOpenAIClient.call_llm_api')
    def test_complete_labeling_workflow(self, mock_api_call):
        """Test complete labeling workflow with mocked API"""
        # Setup mock API response
        mock_api_call.return_value = {
            'content': 'Eiffel Tower Paris Landmark',
            'model': 'mistralai/mistral-7b-instruct',
            'timing': {'llm_response_time': 0.5},
            'tokens': {
                'prompt_tokens': 50,
                'completion_tokens': 10,
                'total_tokens': 60
            }
        }

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            config_manager = ConfigManager(env_path)
            service = LLMLabelingService(config_manager)

            # Test metadata
            metadata = {
                "cluster_id": "test_cluster",
                "image_ids": ["img1", "img2"],
                "text_metadata": ["paris eiffel tower", "eiffel tower night"],
                "cluster_size": 2
            }

            # Generate label
            result = service.generate_cluster_label(metadata)

            # Verify result structure
            assert isinstance(result, LabelResult)
            assert result.label == 'Eiffel Tower Paris Landmark'
            assert result.processing_info['model_used'] == 'mistralai/mistral-7b-instruct'
            assert 'total_processing_time' in result.timing

        finally:
            os.unlink(env_path)

    @patch('src.processing.llm_labelling.DatabaseManager')
    def test_caching_behavior(self, MockDatabaseManager):
        """Test that caching is used correctly"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            config_manager = ConfigManager(env_path)

            # Setup Mock DB Manager instance
            mock_db_instance = MockDatabaseManager.return_value
            mock_db_instance.get_cached_llm_label.return_value = None  # Cache miss first

            service = LLMLabelingService(config_manager)

            # Mock API client to avoid real calls
            service.api_client.call_llm_api = MagicMock(
                return_value={
                    'content': 'Generated Label',
                    'model': 'test-model',
                    'tokens': {
                        'prompt_tokens': 10,
                        'completion_tokens': 5,
                        'total_tokens': 15},
                    'timing': {
                        'llm_response_time': 0.5}})

            input_data = {
                "cluster_id": "test_cluster",
                "image_ids": ["img1"],
                "text_metadata": ["test text"],
                "cluster_size": 1
            }

            # 1. Test Cache Miss
            result = service.generate_cluster_label(input_data)

            assert result.label == 'Generated Label'
            # Verify cache get called
            mock_db_instance.get_cached_llm_label.assert_called_once()
            # Verify cache save called
            mock_db_instance.save_cached_llm_label.assert_called_once()

            # 2. Test Cache Hit
            # Setup mock to return cached data
            mock_db_instance.get_cached_llm_label.return_value = {
                'label': 'Cached Label',
                'processing_info': {},
                'timing': {}
            }
            mock_db_instance.get_cached_llm_label.reset_mock()
            mock_db_instance.save_cached_llm_label.reset_mock()
            service.api_client.call_llm_api.reset_mock()

            result_cached = service.generate_cluster_label(input_data)

            assert result_cached.label == 'Cached Label'
            mock_db_instance.get_cached_llm_label.assert_called_once()
            # Verify API NOT called
            service.api_client.call_llm_api.assert_not_called()

        finally:
            if os.path.exists(env_path):
                os.unlink(env_path)


class TestErrorHandling:
    """Test error handling functionality"""

    def test_error_handling_function(self):
        """Test centralized error handling function"""
        # Test different error types
        from src.processing.llm_labelling import ConfigurationError, InputValidationError, APICommunicationError

        # Test ConfigurationError
        error_info = handle_error(
            ConfigurationError("test error"), {
                "context": "test"})
        assert error_info['error_type'] == 'ConfigurationError'
        assert error_info['recovery_suggestion'] == "Check .env file and API key configuration"
        assert error_info['severity'] == "HIGH"

        # Test InputValidationError
        error_info = handle_error(
            InputValidationError("test error"), {
                "context": "test"})
        assert error_info['error_type'] == 'InputValidationError'
        assert error_info['recovery_suggestion'] == "Validate input metadata structure and content"
        assert error_info['severity'] == "MEDIUM"

        # Test APICommunicationError
        error_info = handle_error(
            APICommunicationError("test error"), {
                "context": "test"})
        assert error_info['error_type'] == 'APICommunicationError'
        assert error_info['recovery_suggestion'] == "Check network connection and retry with exponential backoff"
        assert error_info['severity'] == "HIGH"

        # Test generic error
        error_info = handle_error(Exception("test error"), {"context": "test"})
        assert error_info['error_type'] == 'Exception'
        assert error_info['severity'] == "CRITICAL"


class TestCompleteWorkflow:
    """Test the complete end-to-end workflow"""

    def teardown_method(self):
        """Clean up environment variables after each test"""
        vars_to_clear = [
            'OPENROUTER_API_KEY',
            'DEFAULT_LLM_MODEL',
            'API_TIMEOUT',
            'MAX_RETRIES',
            'LLM_PROVIDER',
            'OLLAMA_BASE_URL',
            'OLLAMA_API_KEY'
        ]
        for var in vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    @patch('src.processing.llm_labelling.UnifiedOpenAIClient.call_llm_api')
    def test_main_workflow_success(self, mock_api_call):
        """Test successful execution of main workflow"""
        # Setup mock API response
        mock_api_call.return_value = {
            'content': 'Beautiful Paris Cityscape',
            'model': 'mistralai/mistral-7b-instruct',
            'timing': {'llm_response_time': 0.3},
            'tokens': {
                'prompt_tokens': 40,
                'completion_tokens': 8,
                'total_tokens': 48
            }
        }

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            # Test metadata
            metadata = {
                "cluster_id": "paris_cluster",
                "image_ids": ["paris_001", "paris_002", "paris_003"],
                "text_metadata": [
                    "paris cityscape with eiffel tower",
                    "eiffel tower at sunset",
                    "paris landmarks and architecture"
                ],
                "cluster_size": 3,
                "spatial_info": {
                    "centroid": [48.8584, 2.2945],
                    "bounding_box": [[48.85, 2.29], [48.86, 2.30]]
                }
            }

            # Execute main workflow
            result = main_llm_labelling_workflow(metadata, env_path)

            # Verify result structure
            assert 'label' in result
            assert 'processing_info' in result
            assert 'timing' in result
            assert result['label'] == 'Beautiful Paris Cityscape'
            assert result['processing_info']['model_used'] == 'mistralai/mistral-7b-instruct'

        finally:
            os.unlink(env_path)

    def test_main_workflow_error_handling(self):
        """Test error handling in main workflow"""
        # Create temporary .env file with invalid API key
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=invalid\n")
            env_path = f.name

        try:
            metadata = {
                "cluster_id": "test_cluster",
                "image_ids": ["img1"],
                "text_metadata": ["test text"],
                "cluster_size": 1
            }

            # Test workflow with invalid configuration
            with pytest.raises(LLMProcessingError) as exc_info:
                main_llm_labelling_workflow(metadata, env_path)

            assert "LLM workflow failed" in str(exc_info.value)

        finally:
            os.unlink(env_path)


class TestSampleImageClusterMetadata:
    """Test with realistic sample image cluster metadata"""

    def teardown_method(self):
        """Clean up environment variables after each test"""
        vars_to_clear = [
            'OPENROUTER_API_KEY',
            'DEFAULT_LLM_MODEL',
            'API_TIMEOUT',
            'MAX_RETRIES',
            'LLM_PROVIDER',
            'OLLAMA_BASE_URL',
            'OLLAMA_API_KEY'
        ]
        for var in vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    @patch('src.processing.llm_labelling.UnifiedOpenAIClient.call_llm_api')
    def test_realistic_cluster_metadata(self, mock_api_call):
        """Test with realistic sample cluster metadata"""
        # Setup mock API response
        mock_api_call.return_value = {
            'content': 'Eiffel Tower Paris Landmark',
            'model': 'mistralai/mistral-7b-instruct',
            'timing': {'llm_response_time': 0.4},
            'tokens': {
                'prompt_tokens': 60,
                'completion_tokens': 12,
                'total_tokens': 72
            }
        }

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-valid_api_key_12345678901234567890\n")
            env_path = f.name

        try:
            # Realistic sample metadata
            realistic_metadata = {
                "cluster_id": "cluster_001",
                "image_ids": ["img_001", "img_002", "img_003", "img_004"],
                "text_metadata": [
                    "paris eiffel tower landmark",
                    "eiffel tower night lights",
                    "paris cityscape with eiffel",
                    "eiffel tower architecture"
                ],
                "cluster_size": 4,
                "spatial_info": {
                    "centroid": [48.8584, 2.2945],
                    "bounding_box": [[48.85, 2.29], [48.86, 2.30]]
                }
            }

            config_manager = ConfigManager(env_path)
            service = LLMLabelingService(config_manager)

            # Test complete workflow
            result = service.generate_cluster_label(realistic_metadata)

            # Verify results
            assert isinstance(result, LabelResult)
            assert result.label == 'Eiffel Tower Paris Landmark'
            assert result.processing_info['model_used'] == 'mistralai/mistral-7b-instruct'

        finally:
            os.unlink(env_path)
