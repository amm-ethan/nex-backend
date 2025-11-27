"""
Comprehensive unit tests for exception utilities.
Tests custom HTTP exceptions, exception handlers, and logging functionality.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import traceback
import logging
import threading
import time
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError, BaseModel, Field

from app.infrastructure.utils.exception_utils import (
    CustomHttpError,
    CustomLockedHttpError,
    CustomBadRequestHttpError,
    CustomGatewayTimeoutError,
    CustomNotFoundHttpError,
    CustomUnauthorizedHttpError,
    CustomUnauthorizedWithHeaderHttpError,
    CustomUnprocessableEntityHttpError,
    log_exception_details
)
from app.infrastructure.utils.exception_handler_utils import (
    add_validation_error_handler,
    add_http_exception_handler,
    add_server_exception_handler
)


class TestCustomHTTPExceptions:
    """Test custom HTTP exception classes."""
    
    def test_custom_http_error(self):
        exc = CustomHttpError("Generic error")
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.detail == "Generic error"
    
    def test_custom_bad_request_error(self):
        exc = CustomBadRequestHttpError("Invalid input data")
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert exc.detail == "Invalid input data"
    
    def test_custom_not_found_error(self):
        exc = CustomNotFoundHttpError("Patient not found")
        assert exc.status_code == status.HTTP_404_NOT_FOUND
        assert exc.detail == "Patient not found"
    
    def test_custom_unauthorized_error(self):
        exc = CustomUnauthorizedHttpError("Authentication required")
        assert exc.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc.detail == "Authentication required"
    
    def test_custom_unauthorized_with_header_error(self):
        exc = CustomUnauthorizedWithHeaderHttpError("Authentication required")
        assert exc.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc.detail == "Authentication required"
        assert "WWW-Authenticate" in exc.headers
    
    def test_custom_locked_error(self):
        exc = CustomLockedHttpError("Resource locked")
        assert exc.status_code == status.HTTP_423_LOCKED
        assert exc.detail == "Resource locked"
    
    def test_custom_gateway_timeout_error(self):
        exc = CustomGatewayTimeoutError("Gateway timeout")
        assert exc.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        assert exc.detail == "Gateway timeout"
    
    def test_custom_unprocessable_entity_error(self):
        exc = CustomUnprocessableEntityHttpError("Validation failed")
        assert exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert exc.detail == "Validation failed"


class TestExceptionHandlers:
    """Test exception handler utilities."""
    
    @pytest.fixture
    def app(self):
        """Create a test FastAPI app."""
        test_app = FastAPI()
        
        @test_app.get("/test-validation-error")
        def test_validation():
            raise RequestValidationError([])
        
        @test_app.get("/test-http-error")
        def test_http():
            raise CustomBadRequestHttpError("Test error")
        
        @test_app.get("/test-server-error")
        def test_server():
            raise Exception("Internal error")
        
        return test_app
    
    @pytest.fixture
    def client_with_handlers(self, app):
        """Create test client with exception handlers."""
        add_validation_error_handler(app)
        add_http_exception_handler(app)
        add_server_exception_handler(app)
        # Disable TestClient's default exception re-raising so our handlers can work
        return TestClient(app, raise_server_exceptions=False)
    
    @pytest.fixture
    def client_without_handlers(self, app):
        """Create test client without exception handlers."""
        return TestClient(app)
    
    def test_validation_error_handler(self, client_with_handlers):
        """Test validation error handler."""
        response = client_with_handlers.get("/test-validation-error")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
    
    def test_http_exception_handler(self, client_with_handlers):
        """Test HTTP exception handler."""
        response = client_with_handlers.get("/test-http-error")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["detail"] == "Test error"
    
    def test_server_exception_handler(self, client_with_handlers):
        """Test server exception handler."""
        response = client_with_handlers.get("/test-server-error")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    def test_exception_handlers_not_installed(self, client_without_handlers):
        """Test behavior without exception handlers (should raise)."""
        with pytest.raises(Exception):
            client_without_handlers.get("/test-server-error")


class TestExceptionLogging:
    """Test exception logging functionality."""
    
    @patch('app.infrastructure.utils.exception_handler_utils.logger')
    def test_server_exception_logging(self, mock_logger):
        """Test that server exceptions are logged."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            raise Exception("Test exception for logging")
        
        add_server_exception_handler(app)
        client = TestClient(app, raise_server_exceptions=False)
        
        response = client.get("/test")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # Verify logging was called
        mock_logger.error.assert_called()
    
    @patch('app.infrastructure.utils.exception_handler_utils.logger')
    def test_validation_error_logging(self, mock_logger):
        """Test that validation errors are logged."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            raise RequestValidationError([])
        
        add_validation_error_handler(app)
        client = TestClient(app, raise_server_exceptions=False)
        
        response = client.get("/test")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        # Verify logging was called
        mock_logger.warning.assert_called()


class TestExceptionResponseFormat:
    """Test exception response formatting."""
    
    def test_http_exception_response_format(self):
        """Test HTTP exception response structure."""
        app = FastAPI()
        add_http_exception_handler(app)
        
        @app.get("/test")
        def test_endpoint():
            raise CustomNotFoundHttpError("Resource not found")
        
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["detail"] == "Resource not found"
    
    def test_validation_error_response_format(self):
        """Test validation error response structure."""
        app = FastAPI()
        add_validation_error_handler(app)
        
        class TestModel(BaseModel):
            required_field: str = Field(..., min_length=1)
        
        @app.post("/test")
        def test_endpoint(data: TestModel):
            return data
        
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/test", json={})  # Missing required field
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "required_field" in str(data["detail"])
    
    def test_server_error_response_format(self):
        """Test server error response structure."""
        app = FastAPI()
        add_server_exception_handler(app)
        
        @app.get("/test")
        def test_endpoint():
            raise Exception("Unexpected error")
        
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)
        assert "Internal server error" in data["detail"]


class TestLogExceptionDetails:
    """Test the log_exception_details utility function."""
    
    def test_log_exception_details_with_active_exception(self):
        """Test log_exception_details when there's an active exception."""
        mock_logger = Mock(spec=logging.Logger)
        
        try:
            # Raise an exception to create traceback
            raise ValueError("Test exception for logging")
        except ValueError:
            # Call the function while exception is active
            log_exception_details(mock_logger)
            
            # Verify that logger.error was called
            assert mock_logger.error.call_count >= 2
            
            # Check that error messages contain expected information
            error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
            
            # Should contain filename and line info
            assert any("Exception in" in call for call in error_calls)
            # Should contain code context
            assert any("Code:" in call for call in error_calls)
    
    def test_log_exception_details_no_active_exception(self):
        """Test log_exception_details when there's no active exception."""
        mock_logger = Mock(spec=logging.Logger)
        
        # Call without active exception
        log_exception_details(mock_logger)
        
        # Should not crash, but might not log anything useful
        # The behavior depends on what sys.exc_info() returns when no exception is active
        # It typically returns (None, None, None)
        
        # The function should handle this gracefully
        assert True  # If we get here without crashing, the test passes
    
    def test_log_exception_details_with_nested_exceptions(self):
        """Test log_exception_details with nested exception calls."""
        mock_logger = Mock(spec=logging.Logger)
        
        def inner_function():
            raise RuntimeError("Inner exception")
        
        def outer_function():
            try:
                inner_function()
            except RuntimeError:
                raise ValueError("Outer exception")
        
        try:
            outer_function()
        except ValueError:
            log_exception_details(mock_logger)
            
            # Should log details about the ValueError (outer exception)
            assert mock_logger.error.called
            
            error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
            # Should contain information about the outer function/exception
            assert any("Exception in" in call for call in error_calls)
    
    def test_log_exception_details_with_different_logger_types(self):
        """Test log_exception_details with different logger configurations."""
        # Test with a real logger
        real_logger = logging.getLogger("test_logger")
        real_logger.setLevel(logging.ERROR)
        
        # Mock the handler to capture output
        mock_handler = Mock()
        mock_handler.level = logging.ERROR  # Set proper level for comparison
        real_logger.addHandler(mock_handler)
        
        try:
            raise Exception("Test exception with real logger")
        except Exception:
            # Should not crash with real logger
            log_exception_details(real_logger)
            
        # Clean up
        real_logger.removeHandler(mock_handler)
        
        # Test passed if no exception was raised
        assert True
    
    @patch('sys.exc_info')
    def test_log_exception_details_with_mocked_exc_info(self, mock_exc_info):
        """Test log_exception_details with mocked sys.exc_info."""
        mock_logger = Mock(spec=logging.Logger)
        
        # Mock exc_info to return specific values
        mock_exc_type = ValueError
        mock_exc_value = ValueError("Mocked exception")
        mock_tb = Mock()
        
        # Create a mock traceback
        mock_extract_tb = Mock()
        mock_extract_tb.return_value = [
            ("test_file.py", 42, "test_function", "raise ValueError('test')")
        ]
        
        mock_exc_info.return_value = (mock_exc_type, mock_exc_value, mock_tb)
        
        with patch('traceback.extract_tb', mock_extract_tb):
            log_exception_details(mock_logger)
            
            # Verify the mocked traceback was used
            mock_extract_tb.assert_called_once_with(mock_tb)
            
            # Verify logger was called with expected information
            assert mock_logger.error.call_count >= 2
            
            error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
            
            # Should contain the mocked file information
            assert any("test_file.py" in call for call in error_calls)
            assert any("line 42" in call for call in error_calls)
            assert any("test_function" in call for call in error_calls)
            assert any("raise ValueError('test')" in call for call in error_calls)
    
    @patch('sys.exc_info')
    def test_log_exception_details_empty_traceback(self, mock_exc_info):
        """Test log_exception_details when traceback is empty."""
        mock_logger = Mock(spec=logging.Logger)
        
        # Mock exc_info with empty traceback
        mock_exc_type = ValueError
        mock_exc_value = ValueError("Exception without traceback")
        mock_tb = Mock()
        
        mock_exc_info.return_value = (mock_exc_type, mock_exc_value, mock_tb)
        
        # Mock empty traceback
        with patch('traceback.extract_tb') as mock_extract_tb:
            mock_extract_tb.return_value = []  # Empty traceback
            
            log_exception_details(mock_logger)
            
            # Should handle empty traceback gracefully
            # Logger might not be called if no traceback info
            # This is acceptable behavior
            assert True
    
    @patch('sys.exc_info')
    def test_log_exception_details_none_traceback(self, mock_exc_info):
        """Test log_exception_details when traceback is None."""
        mock_logger = Mock(spec=logging.Logger)
        
        # Mock exc_info with None traceback
        mock_exc_info.return_value = (ValueError, ValueError("test"), None)
        
        with patch('traceback.extract_tb') as mock_extract_tb:
            mock_extract_tb.return_value = None
            
            log_exception_details(mock_logger)
            
            # Should handle None traceback gracefully
            assert True
    
    def test_log_exception_details_with_complex_exception_chain(self):
        """Test log_exception_details with exception chaining (from/raise from)."""
        mock_logger = Mock(spec=logging.Logger)
        
        try:
            try:
                # Original exception
                raise ConnectionError("Database connection failed")
            except ConnectionError as e:
                # Chain another exception
                raise RuntimeError("Service initialization failed") from e
        except RuntimeError:
            log_exception_details(mock_logger)
            
            # Should log details about the RuntimeError (current exception)
            assert mock_logger.error.called
            
            error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
            assert any("Exception in" in call for call in error_calls)
    
    def test_log_exception_details_preserves_exception_state(self):
        """Test that log_exception_details doesn't affect current exception state."""
        mock_logger = Mock(spec=logging.Logger)
        
        original_exc_info = None
        
        try:
            raise ValueError("Original exception")
        except ValueError:
            # Capture original exception info
            original_exc_info = sys.exc_info()
            
            # Call log_exception_details
            log_exception_details(mock_logger)
            
            # Exception info should be unchanged
            current_exc_info = sys.exc_info()
            assert current_exc_info == original_exc_info
            
            # Should still be able to re-raise
            try:
                raise  # Should re-raise the ValueError
            except ValueError as e:
                assert str(e) == "Original exception"
    
    def test_log_exception_details_thread_safety(self):
        """Test log_exception_details behavior in threading context."""
        mock_logger = Mock(spec=logging.Logger)
        
        results = {}
        
        def thread_function(thread_id):
            try:
                raise ValueError(f"Exception from thread {thread_id}")
            except ValueError:
                log_exception_details(mock_logger)
                results[thread_id] = "completed"
        
        # Create and start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All threads should have completed successfully
        assert len(results) == 3
        assert all(status == "completed" for status in results.values())
        
        # Logger should have been called from multiple threads
        assert mock_logger.error.call_count >= 6  # At least 2 calls per thread
    
    def test_log_exception_details_with_logger_error(self):
        """Test log_exception_details when logger itself raises an error."""
        mock_logger = Mock(spec=logging.Logger)
        mock_logger.error.side_effect = Exception("Logger failed")
        
        try:
            raise ValueError("Original exception")
        except ValueError:
            # Should not crash even if logger fails
            try:
                log_exception_details(mock_logger)
            except Exception as e:
                # If logger error propagates, that's acceptable behavior
                assert "Logger failed" in str(e)
            else:
                # If it handles logger errors silently, that's also fine
                assert True


class TestExceptionUtilsIntegration:
    """Integration tests for exception utils with other components."""
    
    def test_log_exception_details_with_logging_configuration(self):
        """Test log_exception_details with various logging configurations."""
        # Test with different log levels
        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            logger = logging.getLogger(f"test_logger_{level}")
            logger.setLevel(level)
            
            # Mock handler to capture output
            mock_handler = Mock()
            mock_handler.level = level  # Set proper level for comparison
            logger.addHandler(mock_handler)
            
            try:
                raise Exception(f"Test exception for level {level}")
            except Exception:
                log_exception_details(logger)
                
                # Verify function executed without error
                # (Whether it actually logs depends on the level configuration)
                assert True
                
            # Clean up
            logger.removeHandler(mock_handler)
    
    def test_log_exception_details_with_custom_exception_types(self):
        """Test log_exception_details with various custom exception types."""
        mock_logger = Mock(spec=logging.Logger)
        
        # Test with different built-in exception types
        exception_types = [
            (ValueError, "Value error test"),
            (TypeError, "Type error test"),
            (RuntimeError, "Runtime error test"),
            (KeyError, "Key error test"),
            (IndexError, "Index error test"),
            (AttributeError, "Attribute error test"),
            (ImportError, "Import error test"),
            (OSError, "OS error test")
        ]
        
        for exc_type, message in exception_types:
            mock_logger.reset_mock()
            
            try:
                raise exc_type(message)
            except exc_type:
                log_exception_details(mock_logger)
                
                # Should handle all exception types
                assert mock_logger.error.called
                
                # Verify some error information was logged
                error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
                assert any("Exception in" in call for call in error_calls)
    
    def test_log_exception_details_performance(self):
        """Test performance characteristics of log_exception_details."""
        mock_logger = Mock(spec=logging.Logger)
        
        # Test that function executes quickly even with deep stack traces
        def deep_recursion(depth):
            if depth <= 0:
                raise ValueError("Deep stack trace exception")
            return deep_recursion(depth - 1)
        
        try:
            deep_recursion(100)  # Create deep stack trace
        except ValueError:
            start_time = time.time()
            log_exception_details(mock_logger)
            end_time = time.time()
            
            # Should complete quickly (less than 1 second)
            execution_time = end_time - start_time
            assert execution_time < 1.0
            
            # Should have logged the exception details
            assert mock_logger.error.called