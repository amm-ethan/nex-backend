import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.core.config import Settings, settings


class TestSettings:
    """Test the Settings configuration class."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        test_settings = Settings()
        
        assert test_settings.PROJECT_NAME == "NEX Backend"
        assert test_settings.VERSION == "1.0.0"
        assert test_settings.ROOT_PATH == ""
        assert isinstance(test_settings.BACKEND_CORS_ORIGINS, list)
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from environment variable."""
        with patch.dict(os.environ, {"BACKEND_CORS_ORIGINS": '["http://localhost:3000", "https://example.com"]'}, clear=True):
            test_settings = Settings()
            assert len(test_settings.BACKEND_CORS_ORIGINS) == 2
            assert "http://localhost:3000" in test_settings.BACKEND_CORS_ORIGINS
            assert "https://example.com" in test_settings.BACKEND_CORS_ORIGINS
    
    def test_cors_origins_single_value(self):
        """Test CORS origins with single value."""
        with patch.dict(os.environ, {"BACKEND_CORS_ORIGINS": '["http://localhost:3000"]'}, clear=True):
            test_settings = Settings()
            assert test_settings.BACKEND_CORS_ORIGINS == ["http://localhost:3000"]
    
    def test_cors_origins_empty(self):
        """Test CORS origins with empty value."""
        with patch.dict(os.environ, {"BACKEND_CORS_ORIGINS": '[]'}, clear=True):
            test_settings = Settings()
            assert test_settings.BACKEND_CORS_ORIGINS == []
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            "PROJECT_NAME": "Test Project",
            "VERSION": "2.0.0",
            "ROOT_PATH": "/api/v2"
        }):
            test_settings = Settings()
            assert test_settings.PROJECT_NAME == "Test Project"
            assert test_settings.VERSION == "2.0.0"
            assert test_settings.ROOT_PATH == "/api/v2"
    
    def test_settings_model_validation(self):
        """Test Pydantic model validation."""
        # Test that Settings can be created with valid data
        test_data = {
            "PROJECT_NAME": "Valid Project",
            "VERSION": "1.0.0",
            "ROOT_PATH": "/api",
            "BACKEND_CORS_ORIGINS": ["http://localhost:3000"]
        }
        
        test_settings = Settings(**test_data)
        assert test_settings.PROJECT_NAME == "Valid Project"
        assert test_settings.VERSION == "1.0.0"
        assert test_settings.ROOT_PATH == "/api"
        assert test_settings.BACKEND_CORS_ORIGINS == ["http://localhost:3000"]


class TestGlobalSettings:
    """Test the global settings instance."""
    
    def test_settings_singleton(self):
        """Test that settings is properly instantiated."""
        assert settings is not None
        assert isinstance(settings, Settings)
        assert hasattr(settings, 'PROJECT_NAME')
        assert hasattr(settings, 'VERSION')
        assert hasattr(settings, 'ROOT_PATH')
        assert hasattr(settings, 'BACKEND_CORS_ORIGINS')


class TestFilePaths:
    """Test file path configurations."""
    
    def test_directory_paths_exist(self):
        """Test that configured directory paths exist."""
        from app.core.config import directory, error_log_directory, trace_log_directory
        
        # These should be Path objects
        assert isinstance(directory, Path)
        assert isinstance(error_log_directory, Path) 
        assert isinstance(trace_log_directory, Path)
        
        # Directory should exist (project root)
        assert directory.exists()
    
    def test_data_file_paths(self):
        """Test data file path configurations."""
        from app.core.config import microbiology_file, transfers_file
        
        # These should be Path objects pointing to CSV files
        assert isinstance(microbiology_file, Path)
        assert isinstance(transfers_file, Path)
        
        # Check file extensions
        assert microbiology_file.suffix == '.csv'
        assert transfers_file.suffix == '.csv'
        
        # Check filenames
        assert 'microbiology' in microbiology_file.name
        assert 'transfers' in transfers_file.name


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_logger_config_structure(self):
        """Test logger configuration has required structure."""
        from app.core.config import logger_config
        
        assert isinstance(logger_config, dict)
        assert 'version' in logger_config
        assert 'formatters' in logger_config
        assert 'handlers' in logger_config
        assert 'loggers' in logger_config
        
        # Check version is valid
        assert logger_config['version'] == 1
    
    def test_logger_formatters(self):
        """Test logger formatters configuration."""
        from app.core.config import logger_config
        
        formatters = logger_config.get('formatters', {})
        
        # Should have at least one formatter
        assert len(formatters) > 0
        
        # Each formatter should have a format string
        for formatter_name, formatter_config in formatters.items():
            assert 'format' in formatter_config
            assert isinstance(formatter_config['format'], str)
            assert len(formatter_config['format']) > 0
    
    def test_logger_handlers(self):
        """Test logger handlers configuration."""
        from app.core.config import logger_config
        
        handlers = logger_config.get('handlers', {})
        
        # Should have at least one handler
        assert len(handlers) > 0
        
        # Each handler should have required fields
        for handler_name, handler_config in handlers.items():
            assert 'class' in handler_config
            assert 'formatter' in handler_config
    
    def test_root_logger_config(self):
        """Test root logger configuration."""
        from app.core.config import logger_config
        
        # Root logger is configured under 'loggers' with empty key
        loggers = logger_config.get('loggers', {})
        root_config = loggers.get('', {})
        
        assert 'level' in root_config
        assert 'handlers' in root_config
        assert isinstance(root_config['handlers'], list)
        assert len(root_config['handlers']) > 0


class TestEnvironmentLoading:
    """Test environment variable loading."""
    
    def test_dotenv_loading(self):
        """Test that settings can load from environment."""
        # Test that Settings class uses Pydantic settings properly
        from app.core.config import Settings
        
        # Test with environment variables
        with patch.dict(os.environ, {"PROJECT_NAME": "Test Project"}, clear=True):
            test_settings = Settings()
            assert test_settings.PROJECT_NAME == "Test Project"
    
    def test_project_root_detection(self):
        """Test PROJECT_ROOT path detection."""
        from app.core.config import PROJECT_ROOT
        
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
        
        # Should contain key project files
        assert (PROJECT_ROOT / "pyproject.toml").exists()
        assert (PROJECT_ROOT / "app").exists()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_invalid_cors_origins_type(self):
        """Test handling of invalid CORS origins data type."""
        # This should not raise an exception but handle gracefully
        with patch.dict(os.environ, {"BACKEND_CORS_ORIGINS": "not-a-list-or-string"}, clear=True):
            try:
                test_settings = Settings()
                # Should either parse as string or use default
                assert isinstance(test_settings.BACKEND_CORS_ORIGINS, list)
            except Exception as e:
                # If validation fails, it should be a Settings/ValidationError
                assert "Error" in str(type(e)) or "ValidationError" in str(type(e))
    
    def test_empty_project_name(self):
        """Test handling of empty project name."""
        with patch.dict(os.environ, {"PROJECT_NAME": ""}):
            test_settings = Settings()
            # Should accept empty string (not None)
            assert test_settings.PROJECT_NAME == ""
    
    def test_version_format(self):
        """Test version format validation."""
        with patch.dict(os.environ, {"VERSION": "invalid.version"}):
            test_settings = Settings()
            # Should accept any string as version
            assert test_settings.VERSION == "invalid.version"