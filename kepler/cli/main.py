"""
Main CLI application for Kepler framework

Implements the main CLI commands using Typer.
"""

import typer
import subprocess
import json
import time
import getpass
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich import print as rprint

from kepler.core.project import KeplerProject
from kepler.core.config import (
    print_prerequisites_report, 
    detect_gcp_credentials, 
    print_gcp_detection_report
)
from kepler.core.global_config import get_global_config_manager, get_global_config
from kepler.core.library_manager import LibraryManager, create_ai_template, install_unlimited_libraries
from kepler.utils.logging import get_logger, set_verbose
from kepler.utils.exceptions import setup_exception_handler, handle_exception, LibraryManagementError
from kepler.utils.synthetic_data import IndustrialDataGenerator, create_lab_dataset

# Initialize Typer app
app = typer.Typer(
    name="kepler",
    help="Kepler Framework - Simple ML for Industrial Data",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the new Kepler project"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    path: Optional[str] = typer.Option(None, "--path", help="Project directory (default: current directory)")
) -> None:
    """
    Initialize a new Kepler project with configuration templates.
    
    Creates the project structure, configuration files, and documentation.
    """
    logger = get_logger()
    logger.info(f"Initializing Kepler project: {project_name}")
    
    # Determine project path - always create a directory with the project name
    if path:
        base_path = Path(path)
        project_path = base_path / project_name
    else:
        project_path = Path.cwd() / project_name
    
    # Create project directory
    project_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created project directory: {project_path}")
    
    # Initialize project
    project = KeplerProject(str(project_path))
    success = project.initialize(project_name, force=force)
    
    if success:
        logger.info(f"Project {project_name} initialized successfully")
    else:
        logger.error(f"Failed to initialize project {project_name}")
        raise typer.Exit(1)


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: init, show, validate, or path"),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing config")
) -> None:
    """
    Manage global Kepler configuration stored securely outside project directories.
    
    Actions:
        init - Create global configuration template
        show - Display current configuration (sanitized)
        validate - Validate credentials and connectivity
        path - Show path to global configuration file
    """
    from rich.table import Table
    from rich.panel import Panel
    
    logger = get_logger()
    config_manager = get_global_config_manager()
    
    if action == "init":
        rprint("🔧 Initializing global Kepler configuration...")
        
        config_path = config_manager.get_global_config_path()
        
        if config_path.exists() and not force:
            rprint(f"⚠️  Global configuration already exists: [cyan]{config_path}[/cyan]")
            rprint("💡 Use --force to overwrite or 'kepler config show' to view current settings")
            return
        
        try:
            config_manager.create_global_config_template(force=force)
            
            rprint(Panel(
                f"✅ [green]Global configuration template created![/green]\n\n"
                f"📁 Location: [cyan]{config_path}[/cyan]\n"
                f"🔒 Permissions: [green]600 (secure)[/green]\n\n"
                f"📝 [yellow]Next steps:[/yellow]\n"
                f"   1. Edit the configuration file with your credentials\n"
                f"   2. Run 'kepler config validate' to test connectivity\n"
                f"   3. Use 'kepler config show' to verify settings",
                title="🛡️ Global Configuration",
                border_style="green"
            ))
            
            logger.info(f"Global configuration template created: {config_path}")
            
        except Exception as e:
            rprint(f"❌ Failed to create global configuration: {e}")
            logger.error(f"Failed to create global config: {e}")
            raise typer.Exit(1)
    
    elif action == "show":
        rprint("📋 Current global configuration (sensitive values hidden):")
        
        try:
            config = get_global_config()
            
            # Create sanitized display
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Category", style="cyan", no_wrap=True)
            table.add_column("Setting", style="white")
            table.add_column("Value", style="green")
            
            # Splunk settings
            table.add_row("Splunk", "Host", config.splunk.host or "[red]Not set[/red]")
            table.add_row("", "Token", "[green]***configured***[/green]" if config.splunk.token else "[red]Not set[/red]")
            table.add_row("", "HEC Token", "[green]***configured***[/green]" if config.splunk.hec_token else "[red]Not set[/red]")
            table.add_row("", "HEC URL", config.splunk.hec_url or "[yellow]Auto-derived[/yellow]")
            table.add_row("", "Verify SSL", f"[{'green' if config.splunk.verify_ssl else 'yellow'}]{config.splunk.verify_ssl}[/]")
            table.add_row("", "Metrics Index", config.splunk.metrics_index)
            
            # GCP settings
            table.add_row("GCP", "Project ID", config.gcp.project_id or "[red]Not set[/red]")
            table.add_row("", "Region", config.gcp.region)
            table.add_row("", "Service Account", "[green]***configured***[/green]" if config.gcp.service_account_file else "[red]Not set[/red]")
            
            # MLflow settings
            table.add_row("MLflow", "Tracking URI", config.mlflow.tracking_uri or "[yellow]Default (local)[/yellow]")
            
            console.print(table)
            
            # Show configuration file location
            config_path = config_manager.get_global_config_path()
            rprint(f"\n📁 Configuration file: [cyan]{config_path}[/cyan]")
            
        except Exception as e:
            rprint(f"❌ Failed to load configuration: {e}")
            rprint("💡 Run 'kepler config init' to create a configuration file")
            raise typer.Exit(1)
    
    elif action == "validate":
        rprint("🔍 Validating global configuration and connectivity...")
        
        try:
            config_manager = get_global_config_manager()
            credentials = config_manager.validate_credentials_available()
            
            # Create validation results table
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Credential", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="dim")
            
            for cred_name, available in credentials.items():
                status = "✅ Available" if available else "❌ Missing"
                color = "green" if available else "red"
                
                details = ""
                if cred_name == "splunk_token" and not available:
                    details = "Required for Splunk REST API"
                elif cred_name == "splunk_hec_token" and not available:
                    details = "Required for writing to Splunk"
                elif cred_name == "gcp_project_id" and not available:
                    details = "Required for GCP deployment"
                
                table.add_row(
                    cred_name.replace("_", " ").title(),
                    f"[{color}]{status}[/{color}]",
                    details
                )
            
            console.print(table)
            
            # Count missing credentials
            missing_count = sum(1 for available in credentials.values() if not available)
            
            if missing_count == 0:
                rprint("\n🎉 [green]All credentials configured![/green]")
            else:
                rprint(f"\n⚠️  [yellow]{missing_count} credential(s) missing.[/yellow]")
                rprint("💡 Edit your global configuration file to add missing credentials")
            
        except Exception as e:
            rprint(f"❌ Configuration validation failed: {e}")
            raise typer.Exit(1)
    
    elif action == "path":
        config_path = config_manager.get_global_config_path()
        rprint(f"📁 Global configuration file: [cyan]{config_path}[/cyan]")
        
        if config_path.exists():
            rprint("✅ [green]File exists[/green]")
            # Check permissions
            import stat
            file_stat = config_path.stat()
            file_mode = stat.filemode(file_stat.st_mode)
            permissions = oct(file_stat.st_mode)[-3:]
            
            if permissions == "600":
                rprint(f"🔒 [green]Secure permissions: {file_mode}[/green]")
            else:
                rprint(f"⚠️  [yellow]Permissions: {file_mode} (should be 600)[/yellow]")
        else:
            rprint("❌ [red]File does not exist[/red]")
            rprint("💡 Run 'kepler config init' to create it")
    
    else:
        rprint(f"❌ Unknown action: {action}")
        rprint("💡 Valid actions: init, show, validate, path")
        raise typer.Exit(1)


@app.command()
def validate() -> None:
    """
    Validate Kepler prerequisites, configuration, and Splunk connectivity.
    
    Performs comprehensive validation including:
    - Python version and required packages
    - GCP credentials detection
    - Splunk connectivity and authentication
    - Project indexes validation and auto-creation
    """
    from rich.panel import Panel
    from rich.table import Table
    from kepler.connectors.splunk import SplunkConnector
    from kepler.utils.exceptions import SplunkConnectionError
    import os
    from pathlib import Path
    
    logger = get_logger()
    overall_success = True
    
    # ====================================
    # STEP 1: Basic Prerequisites
    # ====================================
    rprint("\n[bold blue]🔍 Step 1: Validating Basic Prerequisites...[/bold blue]\n")
    print_prerequisites_report()
    
    # ====================================
    # STEP 2: GCP Credentials
    # ====================================
    rprint("\n[bold blue]🔍 Step 2: Detecting GCP Credentials...[/bold blue]\n")
    gcp_creds = detect_gcp_credentials()
    print_gcp_detection_report(gcp_creds)
    
    # ====================================
    # STEP 3: Project Configuration
    # ====================================
    rprint("\n[bold blue]🔍 Step 3: Validating Project Configuration...[/bold blue]\n")
    
    # Check if we're in a Kepler project
    if not Path("kepler.yml").exists():
        rprint("❌ [red]Not in a Kepler project directory[/red]")
        rprint("💡 Run 'kepler init <project-name>' to create a new project")
        overall_success = False
        rprint("\n[bold red]⚠️  Validation failed - cannot proceed without project configuration[/bold red]")
        raise typer.Exit(1)
    
    try:
        project = KeplerProject()
        config = project.get_config()
        rprint("✅ [green]Project configuration loaded successfully[/green]")
        
        # Show project info
        rprint(f"📁 Project: [cyan]{config.project_name}[/cyan]")
        rprint(f"📂 Splunk Host: [cyan]{config.splunk.host}[/cyan]")
        rprint(f"📊 Indexes: events=[cyan]{config.splunk.events_index}[/cyan], metrics=[cyan]{config.splunk.metrics_index}[/cyan]")
        
    except Exception as e:
        rprint(f"❌ [red]Project configuration error: {e}[/red]")
        overall_success = False
        rprint("\n[bold red]⚠️  Validation failed - fix configuration and try again[/bold red]")
        raise typer.Exit(1)
    
    # ====================================
    # STEP 4: Splunk Connectivity
    # ====================================
    rprint("\n[bold blue]🔍 Step 4: Testing Splunk Connectivity...[/bold blue]\n")
    
    # Load environment variables
    env_path = Path(".env")
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv()
        rprint("✅ [green]Environment variables loaded from .env[/green]")
    else:
        rprint("⚠️  [yellow]No .env file found - using global configuration only[/yellow]")
    
    # Test connectivity
    try:
        splunk = SplunkConnector(
            host=config.splunk.host,
            token=os.getenv('SPLUNK_TOKEN') or config.splunk.token,
            verify_ssl=config.splunk.verify_ssl,
            timeout=config.splunk.timeout
        )
        
        # Perform health check
        rprint("🔌 Testing Splunk REST API connectivity...")
        health = splunk.health_check()
        
        if health.get('connected', False):
            rprint("✅ [green]Splunk connectivity successful![/green]")
            rprint(f"   📊 Server: [cyan]{health.get('server_name', 'Unknown')}[/cyan]")
            rprint(f"   🏷️  Version: [cyan]{health.get('splunk_version', 'Unknown')}[/cyan]")
            rprint(f"   ⚡ Response Time: [cyan]{health.get('response_time_ms', 0):.1f}ms[/cyan]")
        else:
            rprint(f"❌ [red]Splunk connectivity failed: {health.get('error', 'Unknown error')}[/red]")
            overall_success = False
            
        # Test authentication
        if health.get('connected', False):
            rprint("🔐 Testing Splunk authentication...")
            try:
                auth_valid = splunk.test_authentication()
                if auth_valid:
                    rprint("✅ [green]Splunk authentication successful![/green]")
                else:
                    rprint("❌ [red]Splunk authentication failed[/red]")
                    overall_success = False
            except Exception as e:
                rprint(f"❌ [red]Authentication test failed: {e}[/red]")
                overall_success = False
                
    except Exception as e:
        rprint(f"❌ [red]Splunk connection error: {e}[/red]")
        overall_success = False
        splunk = None
    
    # ====================================
    # STEP 5: Index Validation & Creation
    # ====================================
    if splunk and overall_success:
        rprint("\n[bold blue]🔍 Step 5: Validating Project Indexes...[/bold blue]\n")
        
        try:
            # Validate all project indexes
            project_config = {
                'splunk': {
                    'events_index': config.splunk.events_index,
                    'metrics_index': config.splunk.metrics_index,
                    'default_index': config.splunk.default_index
                }
            }
            
            validation_results = splunk.validate_project_indexes(project_config)
            
            # Create detailed results table
            table = Table(title="📊 Index Validation Results", show_header=True, header_style="bold blue")
            table.add_column("Index", style="cyan", no_wrap=True)
            table.add_column("Type", style="dim", no_wrap=True) 
            table.add_column("Status", no_wrap=True)
            table.add_column("Events", justify="right")
            table.add_column("Size", justify="right")
            table.add_column("Details")
            
            for index_name, details in validation_results['validation_details'].items():
                if details['exists'] and details['accessible']:
                    status = "✅ [green]Ready[/green]"
                    events = str(details['event_count']) if details['event_count'] > 0 else "[dim]No data[/dim]"
                    size = f"{details['size_mb']:.1f} MB" if details['size_mb'] > 0 else "[dim]Empty[/dim]"
                    detail_info = "✅ Accessible" if details['has_data'] else "⚠️ No data yet"
                elif index_name in validation_results['indexes_created']:
                    status = "🆕 [green]Created[/green]"
                    events = "[dim]0[/dim]"
                    size = "[dim]0 MB[/dim]"
                    detail_info = "🎉 Auto-created"
                else:
                    status = "❌ [red]Missing[/red]"
                    events = "[dim]N/A[/dim]"
                    size = "[dim]N/A[/dim]"
                    detail_info = f"Error: {details.get('error', 'Unknown')}"
                
                # Determine index type
                index_type = "📊 Metrics" if "metric" in index_name else "📝 Events"
                
                table.add_row(index_name, index_type, status, events, size, detail_info)
            
            rprint(table)
            
            # Show overall status
            status_color = "green" if validation_results['overall_status'] == 'success' else "yellow" if validation_results['overall_status'] == 'partial' else "red"
            status_icon = "✅" if validation_results['overall_status'] == 'success' else "⚠️" if validation_results['overall_status'] == 'partial' else "❌"
            
            rprint(f"\n{status_icon} [bold {status_color}]Index Validation: {validation_results['overall_status'].upper()}[/bold {status_color}]")
            rprint(f"   📊 {validation_results['indexes_valid']}/{validation_results['indexes_checked']} indexes ready")
            
            if validation_results['indexes_created']:
                rprint(f"   🆕 Auto-created: {', '.join(validation_results['indexes_created'])}")
            
            # Show recommendations
            if validation_results['recommendations']:
                rprint("\n[bold blue]💡 Recommendations:[/bold blue]")
                for rec in validation_results['recommendations']:
                    rprint(f"   • {rec}")
            
            if validation_results['overall_status'] != 'success':
                overall_success = False
            
        except Exception as e:
            rprint(f"❌ [red]Index validation failed: {e}[/red]")
            overall_success = False
    
    # ====================================
    # FINAL SUMMARY
    # ====================================
    rprint("\n" + "="*60)
    
    if overall_success:
        rprint(Panel(
            "🎉 [bold green]All validations passed![/bold green]\n\n"
            "✅ Prerequisites met\n"
            "✅ GCP credentials detected\n" 
            "✅ Splunk connectivity confirmed\n"
            "✅ Project indexes ready\n\n"
            "🚀 [bold]Ready to use Kepler![/bold]\n\n"
            "💡 Next steps:\n"
            "   • Run 'kepler lab generate' to create test data\n"
            "   • Run 'kepler extract \"index=main | head 10\"' to test data extraction\n"
            "   • Run 'kepler train data.csv' to train your first model",
            title="🎯 Validation Summary",
            border_style="green"
        ))
    else:
        rprint(Panel(
            "⚠️  [bold yellow]Some validations failed[/bold yellow]\n\n"
            "Please review the errors above and:\n"
            "   • Fix configuration issues\n"
            "   • Check network connectivity\n"
            "   • Verify credentials\n"
            "   • Run 'kepler validate' again\n\n"
            "💡 For help:\n"
            "   • Check README.md for setup instructions\n"
            "   • Verify .env file has correct tokens\n"
            "   • Ensure Splunk is running and accessible",
            title="⚠️  Validation Results", 
            border_style="yellow"
        ))
        raise typer.Exit(1)


@app.command()
def extract(
    query: str = typer.Argument(..., help="SPL query to execute"),
    output: str = typer.Option("data.csv", "--output", "-o", help="Output file name"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Maximum number of results"),
    earliest: Optional[str] = typer.Option(None, "--earliest", help="Earliest time (e.g., -7d)"),
    latest: Optional[str] = typer.Option(None, "--latest", help="Latest time (e.g., now)")
) -> None:
    """
    Extract data from Splunk using SPL queries.
    
    Connects to Splunk, executes the query, and saves results to CSV.
    """
    from kepler.connectors.splunk import SplunkConnector
    from kepler.utils.exceptions import SplunkConnectionError, DataExtractionError
    import os
    from pathlib import Path
    
    logger = get_logger()
    logger.info(f"Starting data extraction with query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Validate that we're in a Kepler project
    project = KeplerProject()
    if not project.validate_project():
        raise typer.Exit(1)
    
    # Get project configuration
    config = project.get_config()
    if not config:
        rprint("❌ Could not load project configuration")
        raise typer.Exit(1)
    
    try:
        # Initialize Splunk connector
        rprint("🔌 Connecting to Splunk...")
        logger.info(f"Connecting to Splunk at {config.splunk.host}")
        
        connector = SplunkConnector(
            host=config.splunk.host,
            token=config.splunk.token,
            verify_ssl=config.splunk.verify_ssl,
            timeout=config.splunk.timeout
        )
        
        # Test connection
        connector.validate_connection()
        rprint("✅ Connected to Splunk successfully")
        
        # Execute search
        rprint(f"🔍 Executing query: [cyan]{query}[/cyan]")
        
        search_params = {
            'query': query,
            'max_results': limit
        }
        
        if earliest:
            search_params['earliest_time'] = earliest
            rprint(f"   📅 Earliest time: {earliest}")
            
        if latest:
            search_params['latest_time'] = latest  
            rprint(f"   📅 Latest time: {latest}")
        
        rprint(f"   📊 Max results: {limit:,}")
        
        # Get results as DataFrame
        with console.status("Executing search...", spinner="dots"):
            df = connector.search_to_dataframe(**search_params)
        
        if df.empty:
            rprint("⚠️  No results found for the query")
            logger.warning("Search returned no results")
            return
        
        rprint(f"✅ Retrieved {len(df):,} rows with {len(df.columns)} columns")
        logger.info(f"Retrieved data shape: {df.shape}")
        
        # Create output directory if needed
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        rprint(f"💾 Saving results to: [green]{output}[/green]")
        df.to_csv(output, index=False)
        
        # Show preview of data
        rprint("\n📋 Data preview:")
        rprint(f"[dim]Columns: {', '.join(df.columns.tolist())}[/dim]")
        
        # Show first few rows (formatted)
        if len(df) > 0:
            preview_rows = min(3, len(df))
            rprint(f"\n[dim]First {preview_rows} rows:[/dim]")
            
            # Create a simple table preview
            from rich.table import Table
            table = Table(show_header=True, header_style="bold blue")
            
            # Add columns (limit to avoid too wide display)
            cols_to_show = df.columns.tolist()[:6]  # Show max 6 columns
            for col in cols_to_show:
                table.add_column(col, style="cyan", no_wrap=True)
            
            if len(df.columns) > 6:
                table.add_column("...", style="dim")
            
            # Add rows
            for i in range(preview_rows):
                row_data = []
                for col in cols_to_show:
                    value = str(df.iloc[i][col])
                    if len(value) > 20:
                        value = value[:17] + "..."
                    row_data.append(value)
                
                if len(df.columns) > 6:
                    row_data.append("...")
                    
                table.add_row(*row_data)
            
            console.print(table)
        
        rprint(f"\n🎉 Data extraction completed successfully!")
        rprint(f"   📁 File: {output}")
        rprint(f"   📊 Rows: {len(df):,}")
        rprint(f"   📋 Columns: {len(df.columns)}")
        
        logger.info(f"Data extraction completed - saved {len(df)} rows to {output}")
        
    except SplunkConnectionError as e:
        rprint(f"❌ Splunk connection error: {e.message}")
        if e.suggestion:
            rprint(f"💡 {e.suggestion}")
        logger.error(f"Splunk connection failed: {e}")
        raise typer.Exit(1)
        
    except DataExtractionError as e:
        rprint(f"❌ Data extraction error: {e.message}")
        if e.suggestion:
            rprint(f"💡 {e.suggestion}")
        logger.error(f"Data extraction failed: {e}")
        raise typer.Exit(1)
        
    except Exception as e:
        rprint(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
        raise typer.Exit(1)
        
    finally:
        # Clean up connection
        try:
            connector.close()
        except:
            pass


@app.command()
def train(
    data_file: str = typer.Argument(..., help="CSV file with training data"),
    target: str = typer.Option("target", "--target", "-t", help="Target column name"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output model file"),
    algorithm: str = typer.Option("auto", "--algorithm", "-a", help="Algorithm: auto, random_forest, xgboost, pytorch, transformers, etc."),
    test_size: float = typer.Option(0.2, "--test-size", help="Test set size (0.0-1.0)"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for reproducibility"),
    cv: bool = typer.Option(False, "--cv", help="Use cross-validation"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of CV folds"),
    # Deep Learning parameters
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs (Deep Learning)"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size (Deep Learning)"),
    learning_rate: float = typer.Option(0.001, "--learning-rate", help="Learning rate (Deep Learning)"),
    # Generative AI parameters  
    text_column: Optional[str] = typer.Option(None, "--text-column", help="Text column name (Generative AI)"),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Pre-trained model name (Generative AI)"),
    # General parameters
    unified: bool = typer.Option(True, "--unified/--legacy", help="Use unified training API")
) -> None:
    """
    Train an AI model using ANY framework with unified or legacy API.
    
    Supports unlimited AI frameworks:
    - Traditional ML: random_forest, xgboost, lightgbm, catboost
    - Deep Learning: pytorch, tensorflow, keras (with --epochs, --batch-size)
    - Generative AI: transformers, langchain (with --text-column, --model-name)
    - Auto-selection: algorithm=auto (analyzes data and chooses best)
    
    Examples:
      kepler train data.csv --target failure --algorithm auto
      kepler train data.csv --target failure --algorithm xgboost
      kepler train data.csv --target failure --algorithm pytorch --epochs 50
      kepler train text_data.csv --target sentiment --algorithm transformers --text-column review
    """
    import pandas as pd
    from pathlib import Path
    from rich.table import Table
    
    logger = get_logger()
    logger.info(f"Starting AI model training: {algorithm} on {data_file}")
    
    if unified:
        # Use new unified training API (default)
        from kepler.train_unified import train as unified_train, get_algorithm_info
        from kepler.utils.exceptions import ModelTrainingError
    else:
        # Use legacy training API
        from kepler.trainers.base import TrainingConfig
        from kepler.trainers.sklearn_trainers import create_trainer
        from kepler.utils.data_validator import validate_dataframe_for_ml
        from kepler.utils.exceptions import ModelTrainingError, DataExtractionError
    
    try:
        # Validate we're in a Kepler project
        project = KeplerProject()
        if not project.validate_project():
            rprint("❌ Not in a Kepler project directory")
            rprint("💡 Run 'kepler init <project_name>' first")
            raise typer.Exit(1)
        
        # Load and validate data
        rprint(f"📊 Loading training data from: [cyan]{data_file}[/cyan]")
        
        if not Path(data_file).exists():
            rprint(f"❌ Data file not found: {data_file}")
            raise typer.Exit(1)
        
        try:
            df = pd.read_csv(data_file)
            rprint(f"✅ Loaded data: [green]{len(df):,} rows × {len(df.columns)} columns[/green]")
        except Exception as e:
            rprint(f"❌ Failed to load data: {e}")
            raise typer.Exit(1)
        
        # Validate data quality for ML
        rprint("🔍 Validating data quality for ML...")
        with console.status("Analyzing data quality...", spinner="dots"):
            validation_report = validate_dataframe_for_ml(df, target_column=target)
        
        # Show data quality summary
        quality_color = "green" if validation_report.ml_ready else "red"
        rprint(f"📋 Data Quality: [{quality_color}]{validation_report.quality_level.value.upper()}[/{quality_color}]")
        rprint(f"   • Missing data: {validation_report.missing_percentage:.1f}%")
        rprint(f"   • Duplicates: {validation_report.duplicate_percentage:.1f}%") 
        rprint(f"   • Usable rows: {validation_report.estimated_usable_rows:,}")
        
        if not validation_report.ml_ready:
            rprint("⚠️  [yellow]Data quality issues detected:[/yellow]")
            for issue in validation_report.issues[:3]:  # Show top 3 issues
                rprint(f"   • {issue.description}")
            rprint("💡 Consider cleaning your data first or use 'kepler extract' with better queries")
            
            if not typer.confirm("Continue training anyway?"):
                raise typer.Exit(1)
        
        if unified:
            # Use unified training API (default)
            rprint(f"🤖 Using unified training API for [cyan]{algorithm}[/cyan]...")
            
            if algorithm != "auto":
                # Show framework info
                try:
                    framework_info = get_algorithm_info(algorithm)
                    rprint(f"   🔧 Framework: {framework_info['framework']}")
                    rprint(f"   📋 Type: {framework_info['type'].value}")
                except Exception:
                    pass
            
            # Prepare unified parameters
            unified_params = {
                'data': df,
                'target': target,
                'algorithm': algorithm,
                'test_size': test_size,
                'random_state': random_state
            }
            
            # Add framework-specific parameters
            if epochs != 100:  # Non-default epochs
                unified_params['epochs'] = epochs
            if batch_size != 32:  # Non-default batch_size
                unified_params['batch_size'] = batch_size
            if learning_rate != 0.001:  # Non-default learning_rate
                unified_params['learning_rate'] = learning_rate
            if text_column:
                unified_params['text_column'] = text_column
            if model_name:
                unified_params['model_name'] = model_name
            
            # Train model with unified API
            rprint("🚀 Training model with unified API...")
            with console.status("Training in progress...", spinner="dots"):
                model = unified_train(**unified_params)
            
            # Convert to legacy result format for display
            class UnifiedResult:
                def __init__(self, model):
                    self.model = model
                    self.model_path = getattr(model, 'filepath', None)
                    self.metrics = getattr(model, 'performance', {})
                    self.training_time = 0  # TODO: Add timing to unified API
                    self.feature_names = getattr(model, 'feature_columns', [])
                    self.target_name = getattr(model, 'target_column', target)
                    self.model_type = getattr(model, 'model_type', 'unknown')
                    self.hyperparameters = {}
            
            result = UnifiedResult(model)
            
        else:
            # Use legacy training API
            from kepler.utils.data_validator import validate_dataframe_for_ml
            
            # Create training configuration
            config = TrainingConfig(
                algorithm=algorithm,
                target_column=target,
                test_size=test_size,
                random_state=random_state,
                model_output_path=output,
                cross_validation=cv,
                cv_folds=cv_folds
            )
            
            # Create and configure trainer
            rprint(f"🤖 Initializing [cyan]{algorithm}[/cyan] trainer (legacy API)...")
            trainer = create_trainer(algorithm, config)
            
            # Train model
            rprint("🚀 Training model...")
            with console.status("Training in progress...", spinner="dots"):
                result = trainer.train(df)
        
        # Display results
        rprint(f"✅ Training completed in [green]{result.training_time:.2f} seconds[/green]")
        
        if result.model_path:
            rprint(f"💾 Model saved to: [green]{result.model_path}[/green]")
        
        # Show metrics in a nice table
        rprint("\n📊 Model Performance:")
        metrics_table = Table(show_header=True, header_style="bold blue")
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="green", justify="right")
        
        for metric_name, metric_value in result.metrics.items():
            if isinstance(metric_value, float):
                formatted_value = f"{metric_value:.4f}"
            else:
                formatted_value = str(metric_value)
            metrics_table.add_row(metric_name.upper(), formatted_value)
        
        console.print(metrics_table)
        
        # Show model info
        rprint(f"\n🔧 Model Details:")
        rprint(f"   • Type: {result.model_type}")
        rprint(f"   • Features: {len(result.feature_names)} ([dim]{', '.join(result.feature_names[:3])}{'...' if len(result.feature_names) > 3 else ''}[/dim])")
        rprint(f"   • Target: {result.target_name}")
        
        # Show key hyperparameters
        if result.hyperparameters:
            key_params = {k: v for k, v in result.hyperparameters.items() 
                         if k in ['n_estimators', 'max_depth', 'random_state', 'max_iter']}
            if key_params:
                params_str = ', '.join([f"{k}={v}" for k, v in key_params.items()])
                rprint(f"   • Key params: {params_str}")
        
        rprint(f"\n🎉 Model training completed successfully!")
        
        if result.model_path:
            rprint(f"💡 Use '[cyan]kepler deploy {result.model_path}[/cyan]' to deploy your model")
        
        logger.info(f"Model training completed successfully - {result.model_path}")
        
    except ModelTrainingError as e:
        rprint(f"❌ Training error: {e.message}")
        if e.suggestion:
            rprint(f"💡 {e.suggestion}")
        logger.error(f"Model training failed: {e}")
        raise typer.Exit(1)
        
    except Exception as e:
        rprint(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def deploy(
    model_file: str = typer.Argument(..., help="Model file to deploy (.pkl)"),
    name: Optional[str] = typer.Option(None, "--name", help="Service name (default: from config)"),
    environment: str = typer.Option("development", "--env", help="Deployment environment")
) -> None:
    """
    Deploy a trained model to Google Cloud Run.
    
    Creates a containerized API and deploys it to Cloud Run.
    """
    # This will be implemented in Sprint 9
    rprint("[yellow]⚠️  'deploy' command will be implemented in Sprint 9[/yellow]")
    rprint(f"[dim]Would deploy model: {model_file}[/dim]")
    rprint(f"[dim]Service name: {name or 'from config'}[/dim]")
    rprint(f"[dim]Environment: {environment}[/dim]")
    
    # For now, just validate that we're in a Kepler project
    project = KeplerProject()
    if not project.validate_project():
        raise typer.Exit(1)


@app.command()
def predict(
    endpoint: str = typer.Argument(..., help="API endpoint URL"),
    data: str = typer.Argument(..., help="Data to predict (JSON string or file path)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to file")
) -> None:
    """
    Make predictions using a deployed model API.
    
    Sends data to the model endpoint and displays results.
    """
    # This will be implemented in Sprint 10
    rprint("[yellow]⚠️  'predict' command will be implemented in Sprint 10[/yellow]")  
    rprint(f"[dim]Would call endpoint: {endpoint}[/dim]")
    rprint(f"[dim]With data: {data}[/dim]")


# Library Management Commands - Unlimited Python Library Support
@app.command()
def libs(
    action: str = typer.Argument(..., help="Action: install, list, validate, template, deps, lock, optimize, setup-ssh, github, local, create-custom"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="AI template: ml, deep_learning, generative_ai, computer_vision, nlp, full_ai"),
    library: Optional[str] = typer.Option(None, "--library", "-l", help="Library to install (name or URL)"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Library source (GitHub URL, local path, etc.)"),
    upgrade: bool = typer.Option(False, "--upgrade", "-U", help="Upgrade existing libraries"),
    branch: Optional[str] = typer.Option(None, "--branch", help="Git branch for repository installs"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Git tag for repository installs"),
    commit: Optional[str] = typer.Option(None, "--commit", help="Git commit for repository installs"),
    editable: bool = typer.Option(True, "--editable/--no-editable", help="Install in editable mode"),
    author: Optional[str] = typer.Option("Kepler User", "--author", help="Author name for custom libraries")
) -> None:
    """
    Manage unlimited Python libraries for AI and Data Science.
    
    Supports ANY Python library from:
    - PyPI official (sklearn, transformers, pytorch)  
    - GitHub repositories (git+https://github.com/user/repo.git)
    - Private repositories (git+ssh://git@company.com/repo.git)
    - Local custom libraries (-e ./my-lib)
    - Wheel files (./dist/package.whl)
    
    Examples:
      kepler libs template --template generative_ai
      kepler libs install --library transformers
      kepler libs install --library git+https://github.com/research/experimental-ai.git
      kepler libs list
      kepler libs validate
    """
    logger = get_logger()
    
    try:
        lib_manager = LibraryManager(".")
        
        if action == "template":
            if not template:
                rprint("❌ Template required. Available: ml, deep_learning, generative_ai, computer_vision, nlp, full_ai")
                raise typer.Exit(1)
                
            rprint(f"🚀 Creating {template} template...")
            success = lib_manager.create_environment_from_template(template)
            
            if success:
                rprint(f"✅ Created requirements.txt with {template} template")
                rprint("📦 Run 'kepler libs install' to install all libraries")
            else:
                rprint("❌ Failed to create template")
                raise typer.Exit(1)
                
        elif action == "install":
            if library:
                # Install specific library
                from kepler.core.library_manager import LibrarySpec, LibrarySource
                
                if library.startswith('git+'):
                    spec = lib_manager._parse_git_requirement(library)
                elif source:
                    spec = LibrarySpec(name=library, source=LibrarySource.PYPI, version=source)
                else:
                    spec = LibrarySpec(name=library, source=LibrarySource.PYPI)
                
                rprint(f"🚀 Installing {library}...")
                success = lib_manager.install_library(spec, upgrade=upgrade)
                
                if success:
                    rprint(f"✅ Successfully installed {library}")
                else:
                    rprint(f"❌ Failed to install {library}")
                    raise typer.Exit(1)
            else:
                # Install from requirements.txt
                rprint("🚀 Installing all libraries from requirements.txt...")
                results = lib_manager.install_from_requirements()
                
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                
                if successful == total:
                    rprint(f"✅ Successfully installed all {total} libraries")
                else:
                    rprint(f"⚠️ Installed {successful}/{total} libraries")
                    failed = [name for name, success in results.items() if not success]
                    rprint(f"❌ Failed: {', '.join(failed)}")
                    
        elif action == "list":
            rprint("📦 Installed Python Libraries:")
            installed = lib_manager.get_installed_libraries()
            
            if not installed:
                rprint("No libraries found in current environment")
                return
                
            from rich.table import Table
            table = Table(title="Installed Libraries")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green") 
            table.add_column("Source", style="yellow")
            table.add_column("Location", style="dim")
            
            for lib in sorted(installed, key=lambda x: x['name']):
                table.add_row(
                    lib['name'],
                    lib['version'],
                    lib['source'],
                    str(lib['location'])[:50] + "..." if len(str(lib['location'])) > 50 else str(lib['location'])
                )
                
            console.print(table)
            rprint(f"\n📊 Total: {len(installed)} libraries installed")
            
        elif action == "validate":
            rprint("🔍 Validating Python library environment...")
            report = lib_manager.validate_environment()
            
            from rich.table import Table
            table = Table(title="Library Environment Validation")
            table.add_column("Library", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Source", style="yellow")
            
            for lib_name, lib_info in report['libraries'].items():
                status = "✅ OK" if lib_info['installed'] else "❌ MISSING"
                table.add_row(lib_name, status, lib_info['source'])
                
            console.print(table)
            
            # Summary
            total = report['total_libraries']
            successful = report['successful_imports']
            
            if successful == total:
                rprint(f"\n✅ All {total} libraries validated successfully")
            else:
                missing = total - successful
                rprint(f"\n⚠️ {missing}/{total} libraries missing or failed to import")
                rprint(f"Missing: {', '.join(report['missing_libraries'])}")
                
        elif action == "deps":
            rprint("🔍 Analyzing dependency graph...")
            report = lib_manager.resolve_dependencies()
            
            from rich.table import Table
            table = Table(title="Dependency Analysis")
            table.add_column("Library", style="cyan")
            table.add_column("Requested", style="yellow")
            table.add_column("Installed", style="green")
            table.add_column("Status", style="magenta")
            
            for lib_name, version_info in report['resolved_versions'].items():
                status = "✅ OK"
                table.add_row(
                    lib_name,
                    version_info['requested'],
                    version_info['installed'],
                    status
                )
            
            console.print(table)
            
            if report['conflicts']:
                rprint(f"\n⚠️ {len(report['conflicts'])} conflicts detected:")
                for conflict in report['conflicts']:
                    rprint(f"  - {conflict['library']}: wants {conflict['requested']}, has {conflict['installed']}")
            else:
                rprint(f"\n✅ No conflicts detected in {report['total_libraries']} libraries")
            
            if report['recommendations']:
                rprint("\n💡 Recommendations:")
                for rec in report['recommendations']:
                    rprint(f"  - {rec}")
                    
        elif action == "lock":
            rprint("🔒 Creating dependency lock file...")
            lib_manager.create_dependency_lock()
            rprint("✅ Created kepler-lock.txt with exact versions")
            
        elif action == "optimize":
            rprint("🚀 Analyzing production optimization opportunities...")
            report = lib_manager.optimize_for_production()
            
            rprint(f"📊 Total libraries: {report['total_libraries']}")
            
            if report['development_only']:
                rprint(f"\n🛠️ Development-only libraries detected ({len(report['development_only'])}):")
                for item in report['development_only']:
                    rprint(f"  - {item['name']}: {item['reason']}")
            
            if report['alternative_suggestions']:
                rprint(f"\n💡 Size optimization suggestions ({len(report['alternative_suggestions'])}):")
                for suggestion in report['alternative_suggestions']:
                    rprint(f"  - {suggestion['current']} → {suggestion['alternative']}")
                    rprint(f"    Reason: {suggestion['reason']}")
            
            if report['production_ready']:
                rprint("\n✅ Environment is production-ready!")
            else:
                rprint("\n⚠️ Environment needs optimization for production")
                rprint("💡 Run 'kepler libs optimize --create-prod' to generate optimized requirements")
        
        elif action == "setup-ssh":
            from kepler.libs import setup_ssh
            rprint("🔑 Setting up SSH authentication for private repositories...")
            
            success = setup_ssh(test_connection=True)
            if success:
                rprint("✅ [green]SSH authentication configured successfully[/green]")
            else:
                rprint("❌ [red]SSH setup failed[/red]")
                raise typer.Exit(1)
                
        elif action == "github":
            if not source:
                rprint("❌ GitHub URL required. Use --source 'user/repo' or full URL")
                raise typer.Exit(1)
                
            from kepler.libs import install_github
            rprint(f"📦 Installing from GitHub: [blue]{source}[/blue]")
            
            if branch:
                rprint(f"   🌿 Branch: {branch}")
            if tag:
                rprint(f"   🏷️ Tag: {tag}")
            if commit:
                rprint(f"   📝 Commit: {commit}")
            
            success = install_github(source, branch=branch, tag=tag, commit=commit)
            if success:
                rprint("✅ [green]GitHub library installed successfully[/green]")
            else:
                rprint("❌ [red]GitHub installation failed[/red]")
                raise typer.Exit(1)
                
        elif action == "local":
            if not source:
                rprint("❌ Local path required. Use --source './path/to/library'")
                raise typer.Exit(1)
                
            from kepler.libs import install_local
            rprint(f"📂 Installing local library: [blue]{source}[/blue]")
            
            success = install_local(source, editable=editable)
            mode = "editable" if editable else "standard"
            if success:
                rprint(f"✅ [green]Local library installed successfully ({mode} mode)[/green]")
            else:
                rprint("❌ [red]Local installation failed[/red]")
                raise typer.Exit(1)
                
        elif action == "create-custom":
            if not library:
                rprint("❌ Library name required. Use --library 'my-custom-lib'")
                raise typer.Exit(1)
                
            from kepler.libs import create_custom_lib
            rprint(f"🔧 Creating custom library template: [blue]{library}[/blue]")
            
            lib_path = create_custom_lib(library, author)
            rprint(f"✅ [green]Custom library template created: {lib_path}[/green]")
            rprint(f"👤 [blue]Author: {author}[/blue]")
            rprint("\n📋 [yellow]Next steps:[/yellow]")
            rprint(f"   1. Edit your library: [blue]{lib_path}[/blue]")
            rprint(f"   2. Install: [blue]kepler libs local --source {lib_path}[/blue]")
            rprint(f"   3. Import: [blue]import {library.replace('-', '_')}[/blue]")
        
        else:
            rprint(f"❌ Unknown action: {action}")
            rprint("Available actions: template, install, list, validate, deps, lock, optimize, setup-ssh, github, local, create-custom")
            raise typer.Exit(1)
            
    except LibraryManagementError as e:
        rprint(f"❌ Library management error: {e.message}")
        if e.suggestion:
            rprint(f"💡 {e.suggestion}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in libs command: {e}")
        rprint(f"❌ Unexpected error: {e}")
        raise typer.Exit(1)


# Deployment Commands - Cloud Run Integration (Task 6.8)
@app.command()
def deploy(
    model_file: str = typer.Argument(..., help="Path to trained model file (.pkl)"),
    cloud: str = typer.Option("gcp", "--cloud", "-c", help="Cloud provider: gcp"),
    project_id: str = typer.Option(None, "--project", "-p", help="Cloud project ID"),
    service_name: Optional[str] = typer.Option(None, "--service", "-s", help="Service name (auto-generated if None)"),
    region: str = typer.Option("us-central1", "--region", "-r", help="Cloud region"),
    memory: str = typer.Option("1Gi", "--memory", "-m", help="Memory allocation (e.g., 1Gi, 2Gi)"),
    cpu: str = typer.Option("1", "--cpu", help="CPU allocation (e.g., 1, 2)"),
    min_instances: int = typer.Option(0, "--min-instances", help="Minimum instances"),
    max_instances: int = typer.Option(100, "--max-instances", help="Maximum instances"),
    splunk_hec_url: Optional[str] = typer.Option(None, "--splunk-hec-url", help="Splunk HEC URL for predictions"),
    splunk_hec_token: Optional[str] = typer.Option(None, "--splunk-hec-token", help="Splunk HEC token"),
    splunk_index: str = typer.Option("ml_predictions", "--splunk-index", help="Splunk index for predictions"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deployed without deploying"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for deployment to complete")
) -> None:
    """
    Deploy trained model to cloud platform with automatic API generation.
    
    Supports ANY AI framework model:
    - Traditional ML: sklearn, XGBoost, LightGBM models
    - Deep Learning: PyTorch, TensorFlow models  
    - Generative AI: transformers, langchain models
    - Custom frameworks: Any Python model
    
    Examples:
      kepler deploy model.pkl --cloud gcp --project my-ml-project
      kepler deploy xgboost_model.pkl --cloud gcp --project my-project --service predictive-maintenance
      kepler deploy pytorch_model.pkl --cloud gcp --project my-project --memory 2Gi --cpu 2
    """
    import joblib
    from pathlib import Path
    
    logger = get_logger()
    
    try:
        # Validate inputs
        model_path = Path(model_file)
        if not model_path.exists():
            rprint(f"❌ Model file not found: {model_file}")
            raise typer.Exit(1)
        
        if cloud != "gcp":
            rprint(f"❌ Unsupported cloud provider: {cloud}")
            rprint("Supported providers: gcp")
            raise typer.Exit(1)
        
        if not project_id:
            # Try to get from gcloud config
            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    project_id = result.stdout.strip()
                    rprint(f"📋 Using gcloud project: [blue]{project_id}[/blue]")
                else:
                    rprint("❌ No project ID specified and none configured in gcloud")
                    rprint("💡 Use: --project YOUR_PROJECT_ID or run: gcloud config set project YOUR_PROJECT_ID")
                    raise typer.Exit(1)
            except Exception:
                rprint("❌ No project ID specified and gcloud not available")
                rprint("💡 Use: --project YOUR_PROJECT_ID")
                raise typer.Exit(1)
        
        # Load and validate model
        rprint(f"📦 Loading model: [blue]{model_file}[/blue]")
        try:
            model = joblib.load(model_path)
            rprint("✅ Model loaded successfully")
        except Exception as e:
            rprint(f"❌ Failed to load model: {e}")
            rprint("💡 Ensure model was saved with joblib.dump()")
            raise typer.Exit(1)
        
        # Show deployment plan
        if not service_name:
            # Generate service name preview
            model_type = getattr(model, 'model_type', 'unknown')
            algorithm = getattr(model, 'algorithm', 'model')
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            service_name = f"kepler-{algorithm}-{model_type}-{timestamp}".lower().replace('_', '-')
        
        rprint("\n🚀 [bold]Deployment Plan:[/bold]")
        rprint(f"   Cloud: [cyan]{cloud.upper()}[/cyan] (Google Cloud Run)")
        rprint(f"   Project: [blue]{project_id}[/blue]")
        rprint(f"   Region: [green]{region}[/green]")
        rprint(f"   Service: [yellow]{service_name}[/yellow]")
        rprint(f"   Resources: [dim]{memory} memory, {cpu} CPU[/dim]")
        rprint(f"   Scaling: [dim]{min_instances}-{max_instances} instances[/dim]")
        
        if splunk_hec_url:
            rprint(f"   Splunk: [purple]{splunk_index}[/purple] (predictions will be written)")
        
        if dry_run:
            rprint("\n🔍 [yellow]Dry run mode - no actual deployment[/yellow]")
            rprint("✅ Deployment plan validated")
            return
        
        # Confirm deployment
        if not typer.confirm(f"\nDeploy to {cloud.upper()}?"):
            rprint("❌ Deployment cancelled")
            raise typer.Exit(0)
        
        # Prepare deployment configuration
        deployment_config = {
            "project_id": project_id,
            "region": region,
            "service_name": service_name,
            "memory": memory,
            "cpu": cpu,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "environment_variables": {}
        }
        
        # Add Splunk configuration if provided
        if splunk_hec_url:
            deployment_config["environment_variables"].update({
                "SPLUNK_HEC_URL": splunk_hec_url,
                "SPLUNK_HEC_TOKEN": splunk_hec_token or "",
                "SPLUNK_INDEX": splunk_index
            })
        
        # Execute deployment
        rprint(f"\n🚀 Deploying to [cyan]{cloud.upper()}[/cyan]...")
        
        with console.status("Building and deploying...", spinner="dots"):
            from kepler.deploy import to_cloud_run
            
            result = to_cloud_run(model, **deployment_config)
        
        # Show results
        if result["success"]:
            rprint(f"\n✅ [green]Deployment successful![/green]")
            rprint(f"🌐 Service URL: [link]{result['service_url']}[/link]")
            rprint(f"📝 Service Name: [blue]{result['service_name']}[/blue]")
            rprint(f"⏱️  Deployment Time: [dim]{result['deployment_time']:.1f}s[/dim]")
            
            # Show health check results
            if result.get("health_checks"):
                health = result["health_checks"]
                if health["overall_status"] == "healthy":
                    rprint("💚 [green]Health checks: PASSED[/green]")
                else:
                    rprint("💛 [yellow]Health checks: PARTIAL[/yellow]")
                    
            rprint(f"\n📋 [dim]API Endpoints:[/dim]")
            rprint(f"   Docs: [link]{result['service_url']}/docs[/link]")
            rprint(f"   Health: [link]{result['service_url']}/healthz[/link]") 
            rprint(f"   Predict: [link]{result['service_url']}/predict[/link]")
            
        else:
            rprint(f"\n❌ [red]Deployment failed[/red]")
            if "error_message" in result:
                rprint(f"Error: {result['error_message']}")
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error(f"Deployment command failed: {e}")
        rprint(f"❌ Deployment error: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    service_name: str = typer.Argument(..., help="Cloud Run service name"),
    cloud: str = typer.Option("gcp", "--cloud", "-c", help="Cloud provider: gcp"),
    project_id: str = typer.Option(None, "--project", "-p", help="Cloud project ID"),
    region: str = typer.Option("us-central1", "--region", "-r", help="Cloud region")
) -> None:
    """
    Get status of deployed model service.
    
    Shows detailed information about the deployment including:
    - Service health and readiness
    - Traffic allocation and revisions
    - Resource usage and scaling
    - Recent logs and metrics
    
    Examples:
      kepler status my-model-api --project my-ml-project
      kepler status predictive-maintenance --project my-project --region us-west1
    """
    logger = get_logger()
    
    try:
        # Get project ID if not provided
        if not project_id:
            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    project_id = result.stdout.strip()
                else:
                    rprint("❌ No project ID specified and none configured in gcloud")
                    raise typer.Exit(1)
            except Exception:
                rprint("❌ No project ID specified")
                raise typer.Exit(1)
        
        rprint(f"📊 Getting status for: [blue]{service_name}[/blue]")
        
        # Get deployment status
        from kepler.deploy import get_status, validate
        
        with console.status("Checking deployment status...", spinner="dots"):
            status_info = get_status(service_name, project_id, region)
            health_info = validate(service_name, project_id, region)
        
        # Display status information
        rprint(f"\n🎯 [bold]Service: {service_name}[/bold]")
        rprint(f"   URL: [link]{status_info.get('url', 'Not available')}[/link]")
        rprint(f"   Region: [green]{region}[/green]")
        rprint(f"   Ready: {'✅' if status_info.get('ready') else '❌'}")
        rprint(f"   Latest Revision: [dim]{status_info.get('latest_revision', 'Unknown')}[/dim]")
        
        # Health check results
        overall_health = health_info.get("overall_status", "unknown")
        health_icon = "💚" if overall_health == "healthy" else "💛" if overall_health == "partial" else "💔"
        rprint(f"\n{health_icon} [bold]Health Status: {overall_health.upper()}[/bold]")
        
        if "healthz" in health_info:
            healthz = health_info["healthz"]
            rprint(f"   /healthz: {healthz.get('status', 'unknown')} ({healthz.get('response_time_ms', 0):.0f}ms)")
        
        if "readyz" in health_info:
            readyz = health_info["readyz"]
            rprint(f"   /readyz: {readyz.get('status', 'unknown')} ({readyz.get('response_time_ms', 0):.0f}ms)")
        
        # Traffic allocation
        if status_info.get("traffic_allocation"):
            rprint(f"\n🚦 [bold]Traffic Allocation:[/bold]")
            for traffic in status_info["traffic_allocation"]:
                percent = traffic.get("percent", 0)
                revision = traffic.get("revisionName", "Unknown")
                rprint(f"   {percent}% → [dim]{revision}[/dim]")
        
    except Exception as e:
        logger.error(f"Status command failed: {e}")
        rprint(f"❌ Status check error: {e}")
        raise typer.Exit(1)


# AutoML Commands - Intelligent Experimentation (Task 8.8-8.9)
@app.command()
def automl(
    action: str = typer.Argument(..., help="Action: run, compare, optimize, industrial"),
    data_file: str = typer.Option(None, "--data", "-d", help="CSV file with training data"),
    target: str = typer.Option("target", "--target", "-t", help="Target column name"),
    algorithms: Optional[str] = typer.Option("all", "--algorithms", "-a", help="Algorithms to test: all, ml, dl, genai, or comma-separated list"),
    optimization_time: str = typer.Option("30m", "--time", help="Optimization time budget (e.g., 30m, 1h, 2h)"),
    top_n: int = typer.Option(3, "--top-n", "-n", help="Number of top models to report"),
    parallel_jobs: int = typer.Option(2, "--parallel", "-j", help="Number of parallel experiments"),
    industrial_constraints: bool = typer.Option(False, "--industrial", help="Apply industrial constraints"),
    use_case: Optional[str] = typer.Option(None, "--use-case", help="Industrial use case: predictive_maintenance, quality_control, anomaly_detection"),
    promote_best: bool = typer.Option(False, "--promote", help="Promote best model to deployment-ready"),
    output_format: str = typer.Option("table", "--format", help="Output format: table, json, summary")
) -> None:
    """
    Run intelligent AutoML experiments with ranking and optimization.
    
    Automatically:
    - Selects best algorithms based on data characteristics
    - Optimizes hyperparameters with Optuna
    - Engineers features automatically
    - Runs experiments in parallel
    - Ranks models by performance
    - Provides deployment-ready recommendations
    
    Examples:
      kepler automl run --data sensor_data.csv --target failure
      kepler automl run --data data.csv --target failure --time 2h --top-n 5
      kepler automl industrial --data industrial.csv --use-case predictive_maintenance
      kepler automl run --data data.csv --target failure --promote
    """
    import pandas as pd
    from pathlib import Path
    from rich.table import Table
    
    logger = get_logger()
    
    try:
        if not data_file:
            rprint("❌ Data file required for AutoML")
            rprint("💡 Use: kepler automl run --data your_data.csv --target your_target")
            raise typer.Exit(1)
        
        # Validate data file
        data_path = Path(data_file)
        if not data_path.exists():
            rprint(f"❌ Data file not found: {data_file}")
            raise typer.Exit(1)
        
        # Load data
        rprint(f"📊 Loading data: [cyan]{data_file}[/cyan]")
        try:
            df = pd.read_csv(data_path)
            rprint(f"✅ Data loaded: [green]{len(df):,} rows × {len(df.columns)} columns[/green]")
        except Exception as e:
            rprint(f"❌ Failed to load data: {e}")
            raise typer.Exit(1)
        
        # Validate target column
        if target not in df.columns:
            rprint(f"❌ Target column '{target}' not found in data")
            rprint(f"Available columns: {', '.join(df.columns)}")
            raise typer.Exit(1)
        
        rprint(f"🎯 Target column: [yellow]{target}[/yellow]")
        
        # Execute AutoML based on action
        if action == "run":
            _run_automl_experiment(df, target, algorithms, optimization_time, top_n, parallel_jobs, output_format)
            
        elif action == "industrial":
            if not use_case:
                use_case = typer.prompt(
                    "Industrial use case", 
                    type=typer.Choice(["predictive_maintenance", "quality_control", "anomaly_detection"])
                )
            
            _run_industrial_automl(df, target, use_case, optimization_time, output_format)
            
        elif action == "compare":
            _compare_automl_algorithms(df, target, algorithms, output_format)
            
        elif action == "optimize":
            algorithm = typer.prompt("Algorithm to optimize", default="auto")
            _optimize_single_algorithm(df, target, algorithm, optimization_time, output_format)
            
        else:
            rprint(f"❌ Unknown AutoML action: {action}")
            rprint("Available actions: run, industrial, compare, optimize")
            raise typer.Exit(1)
        
        # Promote to deployment if requested
        if promote_best and action in ["run", "industrial"]:
            _promote_best_model_to_deployment()
            
    except Exception as e:
        logger.error(f"AutoML command failed: {e}")
        rprint(f"❌ AutoML error: {e}")
        raise typer.Exit(1)


def _run_automl_experiment(df, target, algorithms, optimization_time, top_n, parallel_jobs, output_format):
    """Run complete AutoML experiment with ranking"""
    rprint(f"🤖 [bold]Running AutoML Experiment[/bold]")
    rprint(f"   Algorithms: [cyan]{algorithms}[/cyan]")
    rprint(f"   Time budget: [yellow]{optimization_time}[/yellow]")
    rprint(f"   Parallel jobs: [blue]{parallel_jobs}[/blue]")
    
    with console.status("Running AutoML experiments...", spinner="dots"):
        from kepler.automl import run_experiment_suite, get_experiment_leaderboard
        
        # Run experiment suite
        experiment_results = run_experiment_suite(
            df,
            target,
            algorithms=algorithms.split(",") if algorithms != "all" else None,
            parallel_jobs=parallel_jobs,
            optimization_budget=optimization_time
        )
    
    # Display results
    if output_format == "json":
        import json
        print(json.dumps(experiment_results, indent=2, default=str))
    else:
        # Show leaderboard
        leaderboard = get_experiment_leaderboard(experiment_results)
        rprint("\n🏆 [bold]AutoML Leaderboard (Top Models):[/bold]")
        rprint(leaderboard)
        
        # Show top-N recommendations
        if experiment_results.get("models"):
            rprint(f"\n📋 [bold]Top {top_n} Recommendations:[/bold]")
            sorted_models = sorted(
                experiment_results["models"].items(),
                key=lambda x: x[1].get("score", 0),
                reverse=True
            )[:top_n]
            
            for i, (algo, result) in enumerate(sorted_models, 1):
                score = result.get("score", 0)
                training_time = result.get("training_time", 0)
                rprint(f"   {i}. [cyan]{algo}[/cyan]: {score:.3f} ({training_time:.1f}s)")


def _run_industrial_automl(df, target, use_case, optimization_time, output_format):
    """Run AutoML with industrial constraints"""
    rprint(f"🏭 [bold]Industrial AutoML: {use_case.replace('_', ' ').title()}[/bold]")
    
    with console.status("Running industrial AutoML...", spinner="dots"):
        from kepler.automl import industrial_automl
        
        result = industrial_automl(
            df,
            target,
            use_case=use_case,
            optimization_budget=optimization_time
        )
    
    # Display results
    if output_format == "json":
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        rprint(f"\n🎯 [bold]Industrial AutoML Results:[/bold]")
        rprint(f"   Best Algorithm: [cyan]{result.get('best_algorithm', 'Unknown')}[/cyan]")
        rprint(f"   Performance Score: [green]{result.get('best_score', 0):.3f}[/green]")
        rprint(f"   Deployment Ready: {'✅' if result.get('deployment_ready') else '❌'}")
        
        if result.get('deployment_ready'):
            rprint(f"   Expected Latency: [blue]{result.get('expected_latency_ms', 'Unknown')}ms[/blue]")
            rprint(f"   Model Size: [blue]{result.get('model_size_mb', 'Unknown')}MB[/blue]")
        else:
            rprint(f"   Issues: [red]{', '.join(result.get('issues', []))}[/red]")


def _compare_automl_algorithms(df, target, algorithms, output_format):
    """Compare multiple AutoML algorithms"""
    rprint(f"⚖️ [bold]Comparing AutoML Algorithms[/bold]")
    
    with console.status("Comparing algorithms...", spinner="dots"):
        from kepler.automl import recommend_algorithms
        
        recommendations = recommend_algorithms(
            df,
            target,
            top_k=5
        )
    
    # Display comparison
    table = Table(title="Algorithm Comparison")
    table.add_column("Rank", style="cyan")
    table.add_column("Algorithm", style="bold")
    table.add_column("Score", style="green")
    table.add_column("Reason", style="dim")
    
    for i, rec in enumerate(recommendations, 1):
        table.add_row(
            str(i),
            rec["algorithm"],
            f"{rec['score']:.3f}",
            rec["reason"]
        )
    
    console.print(table)


def _optimize_single_algorithm(df, target, algorithm, optimization_time, output_format):
    """Optimize hyperparameters for single algorithm"""
    rprint(f"🔧 [bold]Optimizing {algorithm.title()} Hyperparameters[/bold]")
    
    with console.status("Optimizing hyperparameters...", spinner="dots"):
        from kepler.automl import optimize_hyperparameters
        
        optimization_result = optimize_hyperparameters(
            df,
            target,
            algorithm,
            timeout=_parse_time_to_seconds(optimization_time)
        )
    
    # Display results
    rprint(f"\n🎯 [bold]Optimization Results:[/bold]")
    rprint(f"   Algorithm: [cyan]{algorithm}[/cyan]")
    rprint(f"   Best Score: [green]{optimization_result.get('best_score', 0):.4f}[/green]")
    rprint(f"   Best Parameters:")
    
    for param, value in optimization_result.get('best_params', {}).items():
        rprint(f"      {param}: [blue]{value}[/blue]")


def _promote_best_model_to_deployment():
    """Promote best model to deployment-ready status"""
    rprint("\n🚀 [bold]Promoting Best Model to Deployment[/bold]")
    rprint("💡 Use: kepler deploy <best_model>.pkl --cloud gcp --project YOUR_PROJECT")


def _parse_time_to_seconds(time_str: str) -> int:
    """Parse time string to seconds"""
    if time_str.endswith('m'):
        return int(time_str[:-1]) * 60
    elif time_str.endswith('h'):
        return int(time_str[:-1]) * 3600
    elif time_str.endswith('s'):
        return int(time_str[:-1])
    else:
        return int(time_str) * 60  # Default to minutes


# Validation and Setup Commands - Essential Ecosystem Validation (Task 7.6-7.8)
@app.command()
def validate(
    target: str = typer.Argument("ecosystem", help="What to validate: ecosystem, splunk, gcp, prerequisites"),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Attempt automatic fixes for common issues"),
    include_optional: bool = typer.Option(True, "--include-optional/--skip-optional", help="Include optional components (MLflow, DVC)"),
    output_format: str = typer.Option("table", "--format", help="Output format: table, json, summary"),
    save_report: Optional[str] = typer.Option(None, "--save", help="Save report to file")
) -> None:
    """
    Validate Kepler ecosystem components with actionable error messages.
    
    Performs comprehensive validation of:
    - Development prerequisites (Python, libraries, Jupyter)
    - Splunk connectivity and authentication
    - GCP services and permissions
    - MLOps tools availability (optional)
    - End-to-end workflow capability
    
    Examples:
      kepler validate                           # Full ecosystem validation
      kepler validate ecosystem --auto-fix      # With automatic fixes
      kepler validate splunk                    # Splunk only
      kepler validate gcp                       # GCP only
      kepler validate --format json --save validation-report.json
    """
    logger = get_logger()
    
    try:
        rprint(f"🔍 Validating [cyan]{target}[/cyan]...")
        
        if target == "ecosystem":
            from kepler.core.ecosystem_validator import validate_ecosystem
            
            with console.status("Running ecosystem validation...", spinner="dots"):
                report = validate_ecosystem(include_optional=include_optional, auto_fix=auto_fix)
            
            # Display results
            _display_validation_report(report, output_format)
            
            # Save report if requested
            if save_report:
                _save_validation_report(report, save_report)
                rprint(f"📄 Report saved to: [blue]{save_report}[/blue]")
            
            # Exit with appropriate code
            if report.overall_status == ValidationLevel.CRITICAL:
                rprint(f"\n❌ [red]Ecosystem validation failed[/red]")
                raise typer.Exit(1)
            elif report.overall_status == ValidationLevel.WARNING:
                rprint(f"\n⚠️  [yellow]Ecosystem validation completed with warnings[/yellow]")
                raise typer.Exit(0)
            else:
                rprint(f"\n✅ [green]Ecosystem validation successful[/green]")
                raise typer.Exit(0)
                
        elif target == "splunk":
            from kepler.core.ecosystem_validator import validate_splunk
            
            with console.status("Validating Splunk...", spinner="dots"):
                results = validate_splunk()
            
            _display_validation_results(results, "Splunk Validation")
            
        elif target == "gcp":
            from kepler.core.ecosystem_validator import validate_gcp
            
            with console.status("Validating GCP...", spinner="dots"):
                results = validate_gcp()
            
            _display_validation_results(results, "GCP Validation")
            
        elif target == "prerequisites":
            from kepler.core.ecosystem_validator import get_ecosystem_validator
            validator = get_ecosystem_validator()
            
            with console.status("Validating prerequisites...", spinner="dots"):
                results = validator.prerequisites_validator.validate_all()
            
            _display_validation_results(results, "Prerequisites Validation")
            
        else:
            rprint(f"❌ Unknown validation target: {target}")
            rprint("Available targets: ecosystem, splunk, gcp, prerequisites")
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error(f"Validation command failed: {e}")
        rprint(f"❌ Validation error: {e}")
        raise typer.Exit(1)


@app.command()
def setup(
    platform: str = typer.Argument(..., help="Platform to setup: splunk, gcp"),
    interactive: bool = typer.Option(True, "--interactive/--non-interactive", help="Interactive configuration"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Configuration file to use"),
    validate_after: bool = typer.Option(True, "--validate/--no-validate", help="Validate after setup"),
    secure_storage: bool = typer.Option(True, "--secure/--plain", help="Use secure credential storage")
) -> None:
    """
    Guided setup for platform integration with secure credential management.
    
    Provides step-by-step configuration for:
    - Splunk Enterprise (host, authentication, HEC)
    - Google Cloud Platform (authentication, project, APIs)
    - MLOps tools (MLflow, DVC)
    
    Examples:
      kepler setup splunk                       # Interactive Splunk setup
      kepler setup gcp                          # Interactive GCP setup  
      kepler setup splunk --non-interactive     # Automated setup
      kepler setup gcp --config gcp-config.yml # From config file
    """
    logger = get_logger()
    
    try:
        rprint(f"🔧 Setting up [cyan]{platform}[/cyan] integration...")
        
        if platform == "splunk":
            _setup_splunk_platform(interactive, config_file, secure_storage)
            
        elif platform == "gcp":
            _setup_gcp_platform(interactive, config_file, secure_storage)
            
        else:
            rprint(f"❌ Unsupported platform: {platform}")
            rprint("Available platforms: splunk, gcp")
            raise typer.Exit(1)
        
        # Validate setup if requested
        if validate_after:
            rprint(f"\n🔍 Validating {platform} setup...")
            
            if platform == "splunk":
                from kepler.core.ecosystem_validator import validate_splunk
                results = validate_splunk()
                _display_validation_results(results, f"Splunk Setup Validation")
                
            elif platform == "gcp":
                from kepler.core.ecosystem_validator import validate_gcp
                results = validate_gcp()
                _display_validation_results(results, f"GCP Setup Validation")
        
        rprint(f"\n✅ [green]{platform.upper()} setup completed[/green]")
        
    except Exception as e:
        logger.error(f"Setup command failed: {e}")
        rprint(f"❌ Setup error: {e}")
        raise typer.Exit(1)


@app.command()
def diagnose(
    issue_type: str = typer.Argument("auto", help="Issue type: auto, connection, authentication, deployment"),
    platform: Optional[str] = typer.Option(None, "--platform", help="Specific platform: splunk, gcp"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose diagnostic output"),
    fix_suggestions: bool = typer.Option(True, "--suggestions/--no-suggestions", help="Show fix suggestions")
) -> None:
    """
    Intelligent troubleshooting and diagnostic system.
    
    Automatically diagnoses common issues and provides actionable solutions:
    - Connection problems (network, SSL, timeouts)
    - Authentication failures (tokens, credentials, permissions)
    - Deployment issues (GCP setup, API access, resource limits)
    - Library conflicts (dependencies, versions, imports)
    
    Examples:
      kepler diagnose                           # Auto-detect and diagnose issues
      kepler diagnose connection                # Focus on connectivity issues
      kepler diagnose authentication --platform splunk
      kepler diagnose deployment --verbose
    """
    logger = get_logger()
    
    try:
        rprint(f"🔍 Diagnosing [cyan]{issue_type}[/cyan] issues...")
        
        if issue_type == "auto":
            # Auto-detect issues by running validation
            from kepler.core.ecosystem_validator import validate_ecosystem
            
            with console.status("Running diagnostic validation...", spinner="dots"):
                report = validate_ecosystem(include_optional=False, auto_fix=False)
            
            # Analyze issues and provide intelligent diagnosis
            _provide_intelligent_diagnosis(report, platform, verbose, fix_suggestions)
            
        elif issue_type == "connection":
            _diagnose_connection_issues(platform, verbose)
            
        elif issue_type == "authentication": 
            _diagnose_authentication_issues(platform, verbose)
            
        elif issue_type == "deployment":
            _diagnose_deployment_issues(verbose)
            
        else:
            rprint(f"❌ Unknown issue type: {issue_type}")
            rprint("Available types: auto, connection, authentication, deployment")
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error(f"Diagnose command failed: {e}")
        rprint(f"❌ Diagnostic error: {e}")
        raise typer.Exit(1)


def _display_validation_report(report, output_format: str) -> None:
    """Display validation report in specified format"""
    from kepler.core.ecosystem_validator import ValidationLevel
    
    if output_format == "json":
        # JSON output
        import json
        report_dict = {
            "overall_status": report.overall_status.value,
            "success_rate": report.success_rate,
            "total_checks": report.total_checks,
            "results": [
                {
                    "check_name": r.check_name,
                    "success": r.success,
                    "level": r.level.value,
                    "message": r.message,
                    "hint": r.hint
                }
                for r in report.results
            ],
            "recommendations": report.recommendations
        }
        print(json.dumps(report_dict, indent=2))
        return
    
    if output_format == "summary":
        # Summary output
        status_icon = {
            ValidationLevel.SUCCESS: "✅",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.CRITICAL: "❌"
        }
        
        rprint(f"\n{status_icon[report.overall_status]} [bold]Ecosystem Status: {report.overall_status.value.upper()}[/bold]")
        rprint(f"Success Rate: {report.success_rate:.1f}% ({report.successful_checks}/{report.total_checks})")
        
        if report.recommendations:
            rprint("\n📋 [bold]Recommendations:[/bold]")
            for rec in report.recommendations:
                rprint(f"   {rec}")
        return
    
    # Table output (default)
    from rich.table import Table
    
    table = Table(title="Kepler Ecosystem Validation Report")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Message", style="white")
    table.add_column("Hint", style="dim")
    
    for result in report.results:
        status_icon = "✅" if result.success else ("⚠️" if result.level == ValidationLevel.WARNING else "❌")
        status_style = "green" if result.success else ("yellow" if result.level == ValidationLevel.WARNING else "red")
        
        table.add_row(
            result.check_name,
            f"[{status_style}]{status_icon} {result.level.value}[/{status_style}]",
            result.message,
            result.hint or ""
        )
    
    console.print(table)
    
    # Show summary
    status_icon = {
        ValidationLevel.SUCCESS: "✅",
        ValidationLevel.WARNING: "⚠️", 
        ValidationLevel.CRITICAL: "❌"
    }
    
    rprint(f"\n{status_icon[report.overall_status]} [bold]Overall Status: {report.overall_status.value.upper()}[/bold]")
    rprint(f"Validation Time: {report.validation_time:.2f}s")
    rprint(f"Success Rate: {report.success_rate:.1f}% ({report.successful_checks}/{report.total_checks})")
    
    # Show recommendations
    if report.recommendations:
        rprint("\n📋 [bold]Recommendations:[/bold]")
        for rec in report.recommendations:
            rprint(f"   {rec}")


def _display_validation_results(results: List, title: str) -> None:
    """Display list of validation results"""
    from rich.table import Table
    from kepler.core.ecosystem_validator import ValidationLevel
    
    table = Table(title=title)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold") 
    table.add_column("Message", style="white")
    
    for result in results:
        status_icon = "✅" if result.success else ("⚠️" if result.level == ValidationLevel.WARNING else "❌")
        status_style = "green" if result.success else ("yellow" if result.level == ValidationLevel.WARNING else "red")
        
        table.add_row(
            result.check_name,
            f"[{status_style}]{status_icon} {result.level.value}[/{status_style}]",
            result.message
        )
    
    console.print(table)


def _save_validation_report(report, file_path: str) -> None:
    """Save validation report to file"""
    import json
    
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": report.overall_status.value,
        "success_rate": report.success_rate,
        "total_checks": report.total_checks,
        "successful_checks": report.successful_checks,
        "failed_checks": report.failed_checks,
        "warning_checks": report.warning_checks,
        "validation_time": report.validation_time,
        "results": [
            {
                "check_name": r.check_name,
                "category": r.category.value,
                "level": r.level.value,
                "success": r.success,
                "message": r.message,
                "details": r.details,
                "hint": r.hint,
                "auto_fix_available": r.auto_fix_available,
                "auto_fix_command": r.auto_fix_command,
                "context": r.context,
                "timestamp": r.timestamp
            }
            for r in report.results
        ],
        "summary": report.summary,
        "recommendations": report.recommendations
    }
    
    with open(file_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


def _setup_splunk_platform(interactive: bool, config_file: Optional[str], secure_storage: bool) -> None:
    """Setup Splunk platform integration"""
    rprint("🔧 [bold]Splunk Setup[/bold]")
    
    if interactive:
        rprint("\n📋 We'll configure Splunk connectivity step by step...")
        
        # Get Splunk host
        host = typer.prompt("Splunk server URL (e.g., https://splunk.company.com:8089)")
        
        # Get authentication token
        if secure_storage:
            token = getpass.getpass("Splunk authentication token: ")
            
            # Store securely
            from kepler.core.security import store_credential
            store_credential("splunk_token", token)
            rprint("✅ Token stored securely")
        else:
            token = typer.prompt("Splunk authentication token", hide_input=True)
        
        # Get HEC configuration (optional)
        setup_hec = typer.confirm("Configure HTTP Event Collector (HEC) for writing results?", default=True)
        
        hec_token = None
        if setup_hec:
            if secure_storage:
                hec_token = getpass.getpass("HEC token: ")
                store_credential("splunk_hec_token", hec_token)
            else:
                hec_token = typer.prompt("HEC token", hide_input=True)
        
        # SSL verification
        verify_ssl = typer.confirm("Verify SSL certificates?", default=True)
        
        # Create configuration
        config = {
            "splunk": {
                "host": host,
                "token": token if not secure_storage else "stored_securely",
                "hec_token": hec_token if not secure_storage else "stored_securely",
                "verify_ssl": verify_ssl,
                "timeout": 30
            }
        }
        
        # Save configuration
        config_path = Path.home() / ".kepler" / "config.yml"
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        rprint(f"✅ Splunk configuration saved to: [blue]{config_path}[/blue]")
        
    else:
        rprint("Non-interactive setup not yet implemented")
        rprint("Use: kepler setup splunk (without --non-interactive)")


def _setup_gcp_platform(interactive: bool, config_file: Optional[str], secure_storage: bool) -> None:
    """Setup GCP platform integration"""
    rprint("🔧 [bold]GCP Setup[/bold]")
    
    if interactive:
        rprint("\n📋 We'll configure GCP integration step by step...")
        
        # Check if gcloud is available
        try:
            result = subprocess.run(["gcloud", "--version"], capture_output=True, timeout=10)
            if result.returncode != 0:
                rprint("❌ Google Cloud SDK not found")
                rprint("💡 Install from: https://cloud.google.com/sdk/docs/install")
                raise typer.Exit(1)
        except Exception:
            rprint("❌ Google Cloud SDK not available")
            raise typer.Exit(1)
        
        # Check authentication
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True, text=True, timeout=10
            )
            
            if not result.stdout.strip():
                rprint("🔐 No active GCP authentication found")
                if typer.confirm("Run 'gcloud auth login' now?"):
                    subprocess.run(["gcloud", "auth", "login"])
                else:
                    rprint("❌ GCP authentication required")
                    raise typer.Exit(1)
            else:
                active_account = result.stdout.strip()
                rprint(f"✅ Authenticated as: [blue]{active_account}[/blue]")
                
        except Exception as e:
            rprint(f"❌ Authentication check failed: {e}")
            raise typer.Exit(1)
        
        # Get or set project
        current_project = None
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=10
            )
            current_project = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            pass
        
        if current_project:
            use_current = typer.confirm(f"Use current project '{current_project}'?", default=True)
            if not use_current:
                project_id = typer.prompt("GCP Project ID")
                subprocess.run(["gcloud", "config", "set", "project", project_id])
            else:
                project_id = current_project
        else:
            project_id = typer.prompt("GCP Project ID")
            subprocess.run(["gcloud", "config", "set", "project", project_id])
        
        # Enable required APIs
        rprint("🔌 Enabling required APIs...")
        required_apis = [
            "run.googleapis.com",
            "cloudbuild.googleapis.com", 
            "artifactregistry.googleapis.com"
        ]
        
        for api in required_apis:
            rprint(f"   Enabling {api}...")
            try:
                subprocess.run(
                    ["gcloud", "services", "enable", api],
                    capture_output=True, timeout=60
                )
                rprint(f"   ✅ {api}")
            except Exception as e:
                rprint(f"   ❌ Failed to enable {api}: {e}")
        
        # Set default region
        default_region = "us-central1"
        region = typer.prompt("Default region for Cloud Run", default=default_region)
        
        # Save GCP configuration
        gcp_config = {
            "gcp": {
                "project_id": project_id,
                "region": region,
                "apis_enabled": required_apis
            }
        }
        
        config_path = Path.home() / ".kepler" / "config.yml"
        
        # Merge with existing config if it exists
        existing_config = {}
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f) or {}
        
        existing_config.update(gcp_config)
        
        with open(config_path, 'w') as f:
            yaml.dump(existing_config, f, default_flow_style=False)
        
        rprint(f"✅ GCP configuration saved to: [blue]{config_path}[/blue]")
        
    else:
        rprint("Non-interactive GCP setup not yet implemented")
        rprint("Use: kepler setup gcp (without --non-interactive)")


def _provide_intelligent_diagnosis(report, platform: Optional[str], verbose: bool, fix_suggestions: bool) -> None:
    """Provide intelligent diagnosis based on validation report"""
    from kepler.core.ecosystem_validator import ValidationLevel
    
    # Analyze patterns in failures
    critical_issues = [r for r in report.results if r.level == ValidationLevel.CRITICAL and not r.success]
    warning_issues = [r for r in report.results if r.level == ValidationLevel.WARNING and not r.success]
    
    if not critical_issues and not warning_issues:
        rprint("✅ [green]No issues detected - ecosystem is healthy[/green]")
        return
    
    rprint(f"\n🔍 [bold]Diagnostic Analysis:[/bold]")
    
    # Categorize issues
    auth_issues = [r for r in critical_issues if "auth" in r.check_name.lower()]
    connection_issues = [r for r in critical_issues if "connect" in r.check_name.lower()]
    config_issues = [r for r in critical_issues if "config" in r.check_name.lower()]
    
    if auth_issues:
        rprint("🔐 [red]Authentication Issues Detected:[/red]")
        for issue in auth_issues:
            rprint(f"   • {issue.check_name}: {issue.message}")
            if fix_suggestions and issue.hint:
                rprint(f"     💡 {issue.hint}")
    
    if connection_issues:
        rprint("🌐 [red]Connectivity Issues Detected:[/red]")
        for issue in connection_issues:
            rprint(f"   • {issue.check_name}: {issue.message}")
            if fix_suggestions and issue.hint:
                rprint(f"     💡 {issue.hint}")
    
    if config_issues:
        rprint("⚙️ [red]Configuration Issues Detected:[/red]")
        for issue in config_issues:
            rprint(f"   • {issue.check_name}: {issue.message}")
            if fix_suggestions and issue.hint:
                rprint(f"     💡 {issue.hint}")
    
    # Show auto-fixes
    auto_fixable = [r for r in critical_issues + warning_issues if r.auto_fix_available]
    if auto_fixable and fix_suggestions:
        rprint("\n🔧 [yellow]Automatic Fixes Available:[/yellow]")
        for fix in auto_fixable:
            rprint(f"   • {fix.check_name}: [blue]{fix.auto_fix_command}[/blue]")


def _diagnose_connection_issues(platform: Optional[str], verbose: bool) -> None:
    """Diagnose connection-specific issues"""
    rprint("🌐 [bold]Connection Diagnostic[/bold]")
    rprint("This feature will be implemented to diagnose network connectivity issues")


def _diagnose_authentication_issues(platform: Optional[str], verbose: bool) -> None:
    """Diagnose authentication-specific issues"""
    rprint("🔐 [bold]Authentication Diagnostic[/bold]")
    rprint("This feature will be implemented to diagnose authentication issues")


def _diagnose_deployment_issues(verbose: bool) -> None:
    """Diagnose deployment-specific issues"""
    rprint("🚀 [bold]Deployment Diagnostic[/bold]")
    rprint("This feature will be implemented to diagnose deployment issues")


@app.command("version")
def show_version() -> None:
    """Show Kepler version information."""
    from kepler import __version__
    rprint(f"[bold green]Kepler Framework v{__version__}[/bold green]")
    rprint("[dim]AI & Data Science Ecosystem Framework[/dim]")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
) -> None:
    """
    Kepler Framework - Simple ML for Industrial Data
    
    Connect Splunk data to machine learning models and deploy them to Google Cloud.
    """
    # Setup exception handling
    setup_exception_handler()
    
    # Initialize logger
    logger = get_logger()
    
    if version:
        from kepler import __version__
        typer.echo(f"Kepler Framework v{__version__}")
        typer.echo("Simple ML for Industrial Data")
        raise typer.Exit()
    
    # Set up verbose logging if requested
    if verbose:
        set_verbose(True)
        logger.debug("Verbose logging enabled")
    
    logger.info("Kepler CLI initialized")


@app.command()
def lab(
    action: str = typer.Argument(..., help="Action to perform: generate, load, validate"),
    duration_hours: int = typer.Option(24, "--duration", "-d", help="Duration in hours for data generation"),
    sensors: int = typer.Option(10, "--sensors", "-s", help="Number of sensors to simulate"),
    upload_to_splunk: bool = typer.Option(False, "--upload", help="Upload generated data to Splunk"),
    events_index: str = typer.Option("kepler_lab", "--events-index", help="Splunk index for events"),
    metrics_index: str = typer.Option("kepler_metrics", "--metrics-index", help="Splunk index for metrics")
) -> None:
    """
    Laboratory commands for testing Kepler with synthetic data.
    
    Actions:
    - generate: Generate synthetic industrial data
    - load: Load data to Splunk (requires generate first)
    - validate: Validate Splunk connection and indices
    - full: Complete lab setup (generate + load + validate)
    """
    logger = get_logger()
    
    if action == "generate":
        _lab_generate_data(duration_hours, sensors)
    elif action == "load":
        _lab_load_data(events_index, metrics_index)
    elif action == "validate":
        _lab_validate_splunk(events_index, metrics_index)
    elif action == "full":
        _lab_full_setup(duration_hours, sensors, events_index, metrics_index)
    else:
        rprint(f"❌ Unknown action: {action}")
        rprint("💡 Valid actions: generate, load, validate, full")
        raise typer.Exit(1)


def _lab_generate_data(duration_hours: int, sensors: int) -> None:
    """Genera datos sintéticos para el laboratorio."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    rprint(f"\n[bold blue]🧪 Generating synthetic industrial data...[/bold blue]")
    rprint(f"📊 Duration: {duration_hours} hours")
    rprint(f"🔧 Sensors: {sensors}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Generar datos de sensores
        task1 = progress.add_task("Generating sensor data...", total=None)
        generator = IndustrialDataGenerator()
        sensor_data = generator.generate_sensor_metrics(
            duration_hours=duration_hours, 
            num_sensors=sensors
        )
        progress.remove_task(task1)
        
        # Generar datos de producción
        task2 = progress.add_task("Generating production data...", total=None)
        production_data = generator.generate_production_metrics(duration_hours=duration_hours)
        progress.remove_task(task2)
        
        # Generar datos de calidad
        task3 = progress.add_task("Generating quality data...", total=None)
        quality_data = generator.generate_quality_metrics(duration_hours=duration_hours)
        progress.remove_task(task3)
        
        # Guardar datos
        task4 = progress.add_task("Saving data files...", total=None)
        
        # Crear directorio lab si no existe
        lab_dir = Path("lab_data")
        lab_dir.mkdir(exist_ok=True)
        
        # Guardar CSVs
        sensor_data.to_csv(lab_dir / "sensor_data.csv", index=False)
        production_data.to_csv(lab_dir / "production_data.csv", index=False)
        quality_data.to_csv(lab_dir / "quality_data.csv", index=False)
        
        # Guardar eventos para Splunk
        events = generator.export_to_splunk_events(sensor_data)
        metrics = generator.export_to_splunk_metrics(sensor_data)
        
        import json
        with open(lab_dir / "splunk_events.json", 'w') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        
        with open(lab_dir / "splunk_metrics.json", 'w') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + '\n')
        
        progress.remove_task(task4)
    
    rprint(f"\n✅ [bold green]Data generation completed![/bold green]")
    rprint(f"📁 Files saved in: {lab_dir.absolute()}")
    rprint(f"📊 Generated {len(sensor_data)} sensor readings")
    rprint(f"🏭 Generated {len(production_data)} production records")
    rprint(f"🔍 Generated {len(quality_data)} quality measurements")
    rprint(f"📦 Prepared {len(events)} events and {len(metrics)} metrics for Splunk")


def _lab_load_data(events_index: str, metrics_index: str) -> None:
    """Carga datos sintéticos a Splunk."""
    from kepler.connectors.hec import HecWriter
    from kepler.core.project import KeplerProject
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    import json
    
    rprint(f"\n[bold blue]📤 Loading synthetic data to Splunk...[/bold blue]")
    
    # Verificar que existen los archivos de datos
    lab_dir = Path("lab_data")
    events_file = lab_dir / "splunk_events.json"
    metrics_file = lab_dir / "splunk_metrics.json"
    
    if not events_file.exists() or not metrics_file.exists():
        rprint("❌ [bold red]Data files not found![/bold red]")
        rprint("💡 Run `kepler lab generate` first")
        raise typer.Exit(1)
    
    try:
        # Cargar configuración del proyecto
        project = KeplerProject()
        config = project.get_config()
        
        # Verificar configuración HEC
        if not config.splunk.hec_token:
            rprint("❌ [bold red]HEC token not configured![/bold red]")
            rprint("💡 Configure HEC token in kepler.yml or global config")
            raise typer.Exit(1)
        
        # Inicializar HEC Writer
        hec_writer = HecWriter(
            hec_url=config.splunk.hec_url or f"{config.splunk.host}:8088/services/collector",
            token=config.splunk.hec_token
        )
        
        # Verificar conectividad
        rprint("🔍 Testing HEC connectivity...")
        if not hec_writer.health_check():
            rprint("❌ [bold red]HEC health check failed![/bold red]")
            rprint("💡 Check HEC configuration and network connectivity")
            raise typer.Exit(1)
        
        rprint("✅ HEC connectivity confirmed")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            # Cargar eventos
            task1 = progress.add_task("Loading events...", total=None)
            
            with open(events_file, 'r') as f:
                events = [json.loads(line) for line in f]
            
            # Enviar eventos en lotes
            batch_size = 100
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                batch_events = [event['event'] for event in batch]
                
                hec_writer.write_events_batch(
                    events=batch_events,
                    index=events_index,
                    source="kepler_lab",
                    sourcetype="industrial_metrics"
                )
                
                progress.update(task1, advance=len(batch))
            
            progress.remove_task(task1)
            
            # Cargar métricas
            task2 = progress.add_task("Loading metrics...", total=None)
            
            with open(metrics_file, 'r') as f:
                metrics = [json.loads(line) for line in f]
            
            # Enviar métricas en lotes
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i + batch_size]
                
                for metric in batch:
                    hec_writer.write_metric(
                        metric_name="industrial_metrics",
                        value=1,  # Dummy value, real data in fields
                        timestamp=metric['time'],
                        fields=metric['fields'],
                        index=metrics_index
                    )
                
                progress.update(task2, advance=len(batch))
            
            progress.remove_task(task2)
        
        rprint(f"\n✅ [bold green]Data loading completed![/bold green]")
        rprint(f"📊 Loaded {len(events)} events to index: {events_index}")
        rprint(f"📈 Loaded {len(metrics)} metrics to index: {metrics_index}")
        rprint("\n💡 Wait 1-2 minutes for data to be searchable in Splunk")
        
    except Exception as e:
        rprint(f"❌ [bold red]Error loading data: {e}[/bold red]")
        raise typer.Exit(1)


def _lab_validate_splunk(events_index: str, metrics_index: str) -> None:
    """Valida la configuración de Splunk y la presencia de datos."""
    from kepler.connectors.splunk import SplunkConnector
    from kepler.core.project import KeplerProject
    from rich.table import Table
    
    rprint(f"\n[bold blue]🔍 Validating Splunk lab setup...[/bold blue]")
    
    try:
        # Cargar configuración
        project = KeplerProject()
        config = project.get_config()
        
        # Conectar a Splunk
        splunk = SplunkConnector(
            host=config.splunk.host,
            token=config.splunk.token,
            verify_ssl=config.splunk.verify_ssl
        )
        
        # Tabla de resultados
        table = Table(title="Splunk Lab Validation")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="white")
        
        # 1. Conectividad
        try:
            is_healthy = splunk.health_check()
            status = "✅ PASS" if is_healthy else "❌ FAIL"
            table.add_row("Splunk Connectivity", status, "REST API connection")
        except Exception as e:
            table.add_row("Splunk Connectivity", "❌ FAIL", str(e))
        
        # 2. Verificar índices
        indices_to_check = [events_index, metrics_index]
        
        for index in indices_to_check:
            try:
                # Buscar datos en el índice
                query = f'search index={index} | head 1'
                results = splunk.search(query)
                
                if results and len(results) > 0:
                    table.add_row(f"Index: {index}", "✅ HAS DATA", f"Found data in index")
                else:
                    table.add_row(f"Index: {index}", "⚠️ EMPTY", f"Index exists but no data")
            except Exception as e:
                table.add_row(f"Index: {index}", "❌ ERROR", str(e))
        
        # 3. Contar eventos
        try:
            count_query = f'search index={events_index} | stats count'
            count_results = splunk.search(count_query)
            event_count = count_results[0]['count'] if count_results else 0
            table.add_row("Event Count", "ℹ️ INFO", f"{event_count} events in {events_index}")
        except Exception as e:
            table.add_row("Event Count", "❌ ERROR", str(e))
        
        # 4. Verificar métricas
        try:
            metrics_query = f'| mstats count WHERE index={metrics_index}'
            metrics_results = splunk.search_metrics(metrics_query)
            metrics_count = len(metrics_results) if metrics_results else 0
            table.add_row("Metrics Count", "ℹ️ INFO", f"{metrics_count} metric points in {metrics_index}")
        except Exception as e:
            table.add_row("Metrics Count", "❌ ERROR", str(e))
        
        console.print(table)
        
        # Sugerencias
        rprint("\n🔧 [bold blue]Next Steps:[/bold blue]")
        rprint("1. Run example queries:")
        rprint(f"   • `kepler extract 'search index={events_index} | head 10'`")
        rprint(f"   • `kepler extract '| mstats avg(_value) WHERE index={metrics_index} span=1h'`")
        rprint("2. Create Jupyter notebook with SDK examples")
        rprint("3. Train models with: `kepler train lab_data/sensor_data.csv`")
        
    except Exception as e:
        rprint(f"❌ [bold red]Validation failed: {e}[/bold red]")
        raise typer.Exit(1)


def _lab_full_setup(duration_hours: int, sensors: int, events_index: str, metrics_index: str) -> None:
    """Ejecuta el setup completo del laboratorio."""
    rprint("\n[bold blue]🧪 Starting Full Lab Setup...[/bold blue]")
    
    # Paso 1: Generar datos
    _lab_generate_data(duration_hours, sensors)
    
    # Paso 2: Cargar a Splunk
    _lab_load_data(events_index, metrics_index)
    
    # Paso 3: Validar
    _lab_validate_splunk(events_index, metrics_index)
    
    rprint("\n🎉 [bold green]Full lab setup completed successfully![/bold green]")
    rprint("🚀 Your Kepler laboratory is ready for testing!")


# Entry point for the CLI
if __name__ == "__main__":
    app()