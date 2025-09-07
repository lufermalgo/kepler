"""
Main CLI application for Kepler framework

Implements the main CLI commands using Typer.
"""

import typer
from typing import Optional
from pathlib import Path
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
    
    # Determine project path
    if path:
        project_path = Path(path)
        project_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using custom project path: {project_path}")
    else:
        project_path = Path.cwd()
        logger.debug(f"Using current directory: {project_path}")
    
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
    algorithm: str = typer.Option("random_forest", "--algorithm", "-a", help="ML algorithm to use"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test set size (0.0-1.0)"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for reproducibility"),
    cv: bool = typer.Option(False, "--cv", help="Use cross-validation"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of CV folds")
) -> None:
    """
    Train a machine learning model on provided data.
    
    Loads data, trains model, evaluates performance, and saves the trained model.
    """
    from kepler.trainers.base import TrainingConfig
    from kepler.trainers.sklearn_trainers import create_trainer
    from kepler.utils.data_validator import validate_dataframe_for_ml
    from kepler.utils.exceptions import ModelTrainingError, DataExtractionError
    import pandas as pd
    from pathlib import Path
    from rich.table import Table
    
    logger = get_logger()
    logger.info(f"Starting model training: {algorithm} on {data_file}")
    
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
        rprint(f"🤖 Initializing [cyan]{algorithm}[/cyan] trainer...")
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
    action: str = typer.Argument(..., help="Action: install, list, validate, template, deps, lock, optimize"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="AI template: ml, deep_learning, generative_ai, computer_vision, nlp, full_ai"),
    library: Optional[str] = typer.Option(None, "--library", "-l", help="Library to install (name or URL)"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Library source (for custom installs)"),
    upgrade: bool = typer.Option(False, "--upgrade", "-U", help="Upgrade existing libraries")
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
        
        else:
            rprint(f"❌ Unknown action: {action}")
            rprint("Available actions: template, install, list, validate, deps, lock, optimize")
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