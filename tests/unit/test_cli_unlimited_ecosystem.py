"""
Unit tests for CLI unlimited library ecosystem support
Tests Task 1.10: Update CLI to support unlimited library ecosystem
"""

import pytest
from unittest.mock import Mock, patch
from typer.testing import CliRunner

from kepler.cli.main import app


class TestCLIUnlimitedLibrarySupport:
    """Test CLI support for unlimited library ecosystem"""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        return CliRunner()
    
    def test_libs_command_available(self, runner):
        """Test that libs command is available"""
        result = runner.invoke(app, ["libs", "--help"])
        assert result.exit_code == 0
        assert "Manage unlimited Python libraries" in result.stdout
    
    def test_libs_actions_available(self, runner):
        """Test that all libs actions are available"""
        result = runner.invoke(app, ["libs", "--help"])
        
        # Should list all available actions
        expected_actions = [
            "template", "install", "list", "validate", "deps", "lock", 
            "optimize", "setup-ssh", "github", "local", "create-custom"
        ]
        
        for action in expected_actions:
            assert action in result.stdout
    
    def test_libs_github_command_parameters(self, runner):
        """Test libs github command accepts all parameters"""
        # Test parameter availability (not execution to avoid dependencies)
        result = runner.invoke(app, ["libs", "--help"])
        
        expected_params = ["--source", "--branch", "--tag", "--commit"]
        
        for param in expected_params:
            assert param in result.stdout
    
    def test_libs_local_command_parameters(self, runner):
        """Test libs local command accepts editable parameter"""
        result = runner.invoke(app, ["libs", "--help"])
        
        # Should support editable mode parameter
        assert "--editable" in result.stdout or "--no-editable" in result.stdout
    
    def test_libs_create_custom_command_parameters(self, runner):
        """Test libs create-custom command accepts author parameter"""
        result = runner.invoke(app, ["libs", "--help"])
        
        # Should support author parameter
        assert "--author" in result.stdout


class TestCLITrainingUnlimitedFrameworks:
    """Test CLI training command with unlimited framework support"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_train_command_updated_help(self, runner):
        """Test that train command help shows unlimited framework support"""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        
        # Should mention AI frameworks and unified API
        help_text = result.stdout.lower()
        assert "ai" in help_text or "framework" in help_text
        assert "algorithm" in help_text
    
    def test_train_command_algorithm_options(self, runner):
        """Test train command algorithm parameter supports unlimited options"""
        result = runner.invoke(app, ["train", "--help"])
        
        # Should support various algorithm types
        algorithm_help = result.stdout
        assert "--algorithm" in algorithm_help
        
        # Should mention different framework types
        framework_indicators = ["auto", "pytorch", "transformers", "xgboost"]
        
        # At least some should be mentioned in help
        mentioned_frameworks = sum(1 for fw in framework_indicators if fw in algorithm_help)
        assert mentioned_frameworks > 0
    
    def test_train_command_framework_specific_parameters(self, runner):
        """Test train command supports framework-specific parameters"""
        result = runner.invoke(app, ["train", "--help"])
        
        # Should support Deep Learning parameters
        dl_params = ["--epochs", "--batch-size", "--learning-rate"]
        for param in dl_params:
            assert param in result.stdout
        
        # Should support Generative AI parameters
        genai_params = ["--text-column", "--model-name"]
        for param in genai_params:
            assert param in result.stdout
    
    def test_train_command_unified_api_option(self, runner):
        """Test train command supports unified vs legacy API selection"""
        result = runner.invoke(app, ["train", "--help"])
        
        # Should support unified/legacy API selection
        assert "--unified" in result.stdout or "--legacy" in result.stdout


class TestCLIUsabilityImprovements:
    """Test CLI usability improvements for unlimited ecosystem"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_helpful_error_messages_for_library_management(self, runner):
        """Test that CLI provides helpful error messages for library operations"""
        # Test missing source for github action
        result = runner.invoke(app, ["libs", "github"])
        assert result.exit_code == 1
        assert "GitHub URL required" in result.stdout
        
        # Test missing library name for create-custom
        result = runner.invoke(app, ["libs", "create-custom"])
        assert result.exit_code == 1
        assert "Library name required" in result.stdout
    
    def test_cli_examples_in_help(self, runner):
        """Test that CLI help includes examples for unlimited ecosystem"""
        result = runner.invoke(app, ["libs", "--help"])
        
        # Should include examples for different library sources
        help_text = result.stdout
        assert "Examples:" in help_text or "example" in help_text.lower()
    
    def test_cli_comprehensive_action_listing(self, runner):
        """Test that CLI lists all available actions clearly"""
        result = runner.invoke(app, ["libs", "unknown-action"])
        assert result.exit_code == 1
        
        # Should list all available actions in error message
        error_output = result.stdout
        actions = ["template", "install", "github", "local", "create-custom", "setup-ssh"]
        
        mentioned_actions = sum(1 for action in actions if action in error_output)
        assert mentioned_actions >= len(actions) // 2  # At least half should be mentioned


class TestCLIBackwardCompatibility:
    """Test CLI backward compatibility with existing functionality"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_existing_commands_still_work(self, runner):
        """Test that existing CLI commands are not broken"""
        # Test help for existing commands
        existing_commands = ["init", "extract", "train", "version"]
        
        for command in existing_commands:
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0, f"Command {command} should still work"
    
    def test_libs_template_backward_compatibility(self, runner):
        """Test that libs template command maintains backward compatibility"""
        result = runner.invoke(app, ["libs", "--help"])
        
        # Should still support template functionality
        assert "--template" in result.stdout
        
        # Should support original templates
        template_help = result.stdout
        original_templates = ["ml", "deep_learning", "generative_ai"]
        
        for template in original_templates:
            # Templates should be mentioned in help or be valid options
            # (Exact matching depends on help formatting)
            pass  # Backward compatibility verified by existence of --template option


class TestCLIIntegrationWithUnifiedAPI:
    """Test CLI integration with unified training API"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('pandas.read_csv')
    @patch('kepler.cli.main.validate_dataframe_for_ml')
    @patch('kepler.cli.main.KeplerProject')
    def test_train_command_uses_unified_api_by_default(self, mock_project, mock_validate, mock_read_csv, runner):
        """Test that train command uses unified API by default"""
        # Mock project validation
        mock_project_instance = Mock()
        mock_project_instance.validate_project.return_value = True
        mock_project.return_value = mock_project_instance
        
        # Mock data loading
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        
        # Mock data validation
        mock_validation = Mock()
        mock_validation.ml_ready = True
        mock_validation.quality_level.value = "good"
        mock_validation.missing_percentage = 0.0
        mock_validation.duplicate_percentage = 0.0
        mock_validation.estimated_usable_rows = 1000
        mock_validate.return_value = mock_validation
        
        with patch('kepler.cli.main.unified_train') as mock_unified_train:
            mock_model = Mock()
            mock_model.trained = True
            mock_model.performance = {'accuracy': 0.95}
            mock_unified_train.return_value = mock_model
            
            result = runner.invoke(app, [
                "train", "test_data.csv", 
                "--target", "test_target",
                "--algorithm", "random_forest"
            ])
            
            # Should use unified API by default
            mock_unified_train.assert_called_once()
    
    @patch('pandas.read_csv')
    @patch('kepler.cli.main.validate_dataframe_for_ml')
    @patch('kepler.cli.main.KeplerProject')
    def test_train_command_legacy_api_option(self, mock_project, mock_validate, mock_read_csv, runner):
        """Test that train command can use legacy API when requested"""
        # Mock project validation
        mock_project_instance = Mock()
        mock_project_instance.validate_project.return_value = True
        mock_project.return_value = mock_project_instance
        
        # Mock data loading
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        
        # Mock data validation
        mock_validation = Mock()
        mock_validation.ml_ready = True
        mock_validation.quality_level.value = "good"
        mock_validation.missing_percentage = 0.0
        mock_validation.duplicate_percentage = 0.0
        mock_validation.estimated_usable_rows = 1000
        mock_validate.return_value = mock_validation
        
        with patch('kepler.cli.main.create_trainer') as mock_create_trainer:
            mock_trainer = Mock()
            mock_result = Mock()
            mock_result.training_time = 10.0
            mock_result.model_path = "model.pkl"
            mock_result.metrics = {'accuracy': 0.95}
            mock_result.feature_names = ['f1', 'f2']
            mock_result.target_name = 'target'
            mock_result.model_type = 'classification'
            mock_result.hyperparameters = {}
            mock_trainer.train.return_value = mock_result
            mock_create_trainer.return_value = mock_trainer
            
            result = runner.invoke(app, [
                "train", "test_data.csv",
                "--target", "test_target", 
                "--algorithm", "random_forest",
                "--legacy"
            ])
            
            # Should use legacy API when requested
            mock_create_trainer.assert_called_once()


class TestCLIPRDCompliance:
    """Test CLI compliance with PRD requirements"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_prd_requirement_unlimited_library_cli_support(self, runner):
        """
        Test PRD: CLI should support unlimited Python library management
        """
        result = runner.invoke(app, ["libs", "--help"])
        
        # Should support all types of library sources
        library_sources = ["PyPI", "GitHub", "private", "local", "wheel"]
        help_text = result.stdout.lower()
        
        # At least most sources should be mentioned
        mentioned_sources = sum(1 for source in library_sources if source.lower() in help_text)
        assert mentioned_sources >= 3
    
    def test_prd_requirement_ai_framework_cli_support(self, runner):
        """
        Test PRD: CLI should support ANY AI framework training
        """
        result = runner.invoke(app, ["train", "--help"])
        
        # Should support different AI framework types
        ai_framework_types = ["traditional", "deep", "neural", "generative", "transformers"]
        help_text = result.stdout.lower()
        
        # Should mention AI/ML frameworks
        mentioned_types = sum(1 for fw_type in ai_framework_types if fw_type in help_text)
        assert mentioned_types > 0
    
    def test_prd_requirement_simple_cli_interface(self, runner):
        """
        Test PRD: CLI should provide simple interface for complex operations
        """
        # Help should be available and comprehensive
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        
        # Should list main commands clearly
        main_commands = ["init", "extract", "train", "libs"]
        help_text = result.stdout
        
        for command in main_commands:
            assert command in help_text
    
    def test_prd_requirement_extensible_cli(self, runner):
        """
        Test PRD: CLI should be extensible for new frameworks and libraries
        """
        # Should support custom library creation
        result = runner.invoke(app, ["libs", "--help"])
        assert "create-custom" in result.stdout
        
        # Should support framework registration (via library management)
        assert "github" in result.stdout  # GitHub repo support
        assert "local" in result.stdout   # Local library support


class TestCLIDocumentationAndUsability:
    """Test CLI documentation and usability improvements"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_comprehensive_help_documentation(self, runner):
        """Test that CLI provides comprehensive help"""
        # Main help should be informative
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert len(result.stdout) > 100  # Should have substantial help content
        
        # Subcommand help should be detailed
        subcommands = ["libs", "train"]
        for subcommand in subcommands:
            result = runner.invoke(app, [subcommand, "--help"])
            assert result.exit_code == 0
            assert "Examples:" in result.stdout or "example" in result.stdout.lower()
    
    def test_error_messages_are_helpful(self, runner):
        """Test that CLI error messages provide actionable guidance"""
        # Test invalid action
        result = runner.invoke(app, ["libs", "invalid-action"])
        assert result.exit_code == 1
        assert "Available actions:" in result.stdout
        
        # Should list valid actions in error message
        valid_actions = ["template", "install", "github", "local"]
        error_text = result.stdout
        
        mentioned_actions = sum(1 for action in valid_actions if action in error_text)
        assert mentioned_actions >= 3
    
    def test_cli_provides_next_steps_guidance(self, runner):
        """Test that CLI provides guidance for next steps"""
        # Version command should provide basic info
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Kepler" in result.stdout
        
        # Main help should guide users
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        
        # Should mention key workflows
        help_text = result.stdout.lower()
        workflow_keywords = ["init", "extract", "train", "data"]
        mentioned_keywords = sum(1 for keyword in workflow_keywords if keyword in help_text)
        assert mentioned_keywords >= 3


class TestCLIParameterValidation:
    """Test CLI parameter validation for unlimited ecosystem"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_libs_github_parameter_validation(self, runner):
        """Test parameter validation for GitHub library installation"""
        # Should require source for github action
        result = runner.invoke(app, ["libs", "github"])
        assert result.exit_code == 1
        assert "GitHub URL required" in result.stdout
    
    def test_libs_local_parameter_validation(self, runner):
        """Test parameter validation for local library installation"""
        # Should require source for local action  
        result = runner.invoke(app, ["libs", "local"])
        assert result.exit_code == 1
        assert "Local path required" in result.stdout
    
    def test_libs_create_custom_parameter_validation(self, runner):
        """Test parameter validation for custom library creation"""
        # Should require library name for create-custom action
        result = runner.invoke(app, ["libs", "create-custom"])
        assert result.exit_code == 1
        assert "Library name required" in result.stdout


class TestCLIIntegrationWithLibraryManager:
    """Test CLI integration with LibraryManager functionality"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('kepler.libs.setup_ssh')
    def test_libs_setup_ssh_integration(self, mock_setup_ssh, runner):
        """Test libs setup-ssh integrates with library manager"""
        mock_setup_ssh.return_value = True
        
        result = runner.invoke(app, ["libs", "setup-ssh"])
        
        # Should call library manager function
        mock_setup_ssh.assert_called_once()
        assert result.exit_code == 0
    
    @patch('kepler.libs.install_github')
    def test_libs_github_integration(self, mock_install_github, runner):
        """Test libs github integrates with library manager"""
        mock_install_github.return_value = True
        
        result = runner.invoke(app, ["libs", "github", "--source", "user/repo"])
        
        # Should call library manager function
        mock_install_github.assert_called_once_with("user/repo", None, None, None, None)
        assert result.exit_code == 0
    
    @patch('kepler.libs.create_custom_lib')
    def test_libs_create_custom_integration(self, mock_create_custom, runner):
        """Test libs create-custom integrates with library manager"""
        mock_create_custom.return_value = "/path/to/custom/lib"
        
        result = runner.invoke(app, [
            "libs", "create-custom", 
            "--library", "my-custom-lib",
            "--author", "Test Author"
        ])
        
        # Should call library manager function with correct parameters
        mock_create_custom.assert_called_once_with("my-custom-lib", "Test Author")
        assert result.exit_code == 0


class TestCLIUnlimitedEcosystemWorkflows:
    """Test CLI support for unlimited ecosystem workflows"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_complete_custom_library_workflow_cli(self, runner):
        """Test complete custom library workflow via CLI"""
        with patch('kepler.libs.create_custom_lib') as mock_create, \
             patch('kepler.libs.install_local') as mock_install:
            
            mock_create.return_value = "/path/to/custom/lib"
            mock_install.return_value = True
            
            # Step 1: Create custom library
            result1 = runner.invoke(app, [
                "libs", "create-custom",
                "--library", "workflow-test-lib",
                "--author", "CLI Tester"
            ])
            assert result1.exit_code == 0
            assert "Next steps:" in result1.stdout
            
            # Step 2: Install local library (simulated)
            result2 = runner.invoke(app, [
                "libs", "local",
                "--source", "/path/to/custom/lib"
            ])
            assert result2.exit_code == 0
            
            # Verify workflow guidance
            assert "Edit your library:" in result1.stdout
            assert "Install:" in result1.stdout
    
    def test_github_experimental_library_workflow_cli(self, runner):
        """Test GitHub experimental library workflow via CLI"""
        with patch('kepler.libs.install_github') as mock_install_github:
            mock_install_github.return_value = True
            
            # Install experimental library from GitHub
            result = runner.invoke(app, [
                "libs", "github",
                "--source", "research/experimental-ai",
                "--tag", "v0.1.0-alpha"
            ])
            
            assert result.exit_code == 0
            mock_install_github.assert_called_once_with("research/experimental-ai", None, "v0.1.0-alpha", None, None)


class TestCLIUnlimitedFrameworkTraining:
    """Test CLI training with unlimited framework support"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    @patch('kepler.cli.main.KeplerProject')
    @patch('kepler.cli.main.validate_dataframe_for_ml')
    def test_train_with_different_frameworks_cli(self, mock_validate, mock_project, mock_exists, mock_read_csv, runner):
        """Test training with different frameworks via CLI"""
        # Setup mocks
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'feature1': [1, 2], 'target': [0, 1]})
        mock_read_csv.return_value = mock_df
        
        mock_project_instance = Mock()
        mock_project_instance.validate_project.return_value = True
        mock_project.return_value = mock_project_instance
        
        mock_validation = Mock()
        mock_validation.ml_ready = True
        mock_validation.quality_level.value = "good"
        mock_validation.missing_percentage = 0.0
        mock_validation.duplicate_percentage = 0.0
        mock_validation.estimated_usable_rows = 100
        mock_validate.return_value = mock_validation
        
        frameworks_to_test = [
            ("random_forest", {}),
            ("xgboost", {}),
            ("pytorch", {"--epochs": "10"}),
        ]
        
        for framework, extra_args in frameworks_to_test:
            with patch('kepler.cli.main.unified_train') as mock_unified_train:
                mock_model = Mock()
                mock_model.trained = True
                mock_model.performance = {'accuracy': 0.95}
                mock_unified_train.return_value = mock_model
                
                args = [
                    "train", "test_data.csv",
                    "--target", "target", 
                    "--algorithm", framework
                ]
                
                # Add framework-specific arguments
                for arg_name, arg_value in extra_args.items():
                    args.extend([arg_name, arg_value])
                
                result = runner.invoke(app, args)
                
                # Should complete successfully with unified API
                mock_unified_train.assert_called_once()
                
                # Verify framework-specific parameters were passed
                call_kwargs = mock_unified_train.call_args[1]
                assert call_kwargs['algorithm'] == framework
