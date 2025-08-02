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
from kepler.core.config import print_prerequisites_report
from kepler.core.global_config import get_global_config_manager, get_global_config
from kepler.utils.logging import get_logger, set_verbose
from kepler.utils.exceptions import setup_exception_handler, handle_exception

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
    Validate Kepler prerequisites and configuration.
    
    Checks Python version, required packages, configuration validity,
    and connectivity to external services.
    """
    rprint("\n[bold blue]🔍 Validating Kepler Prerequisites...[/bold blue]\n")
    print_prerequisites_report()


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


@app.command("version")
def show_version() -> None:
    """Show Kepler version information."""
    from kepler import __version__
    rprint(f"[bold green]Kepler Framework v{__version__}[/bold green]")
    rprint("[dim]Simple ML for Industrial Data[/dim]")


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


# Entry point for the CLI
if __name__ == "__main__":
    app()