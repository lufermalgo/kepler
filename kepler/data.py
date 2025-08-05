"""
Kepler Data Module - Simple data extraction for data scientists
"""

import pandas as pd
from typing import Optional, Union, Dict, Any
from datetime import datetime, timedelta
import os


def from_splunk(
    spl: str = None,
    index: str = None, 
    query_type: str = "auto",  # "auto", "events", "metrics", "custom"
    time_range: str = "-24h",
    earliest: str = None,
    latest: str = None,
    limit: int = 10000
) -> pd.DataFrame:
    """
    Extract data from Splunk using flexible SPL queries.
    
    The scientist/analyst defines the SPL query they want, and Kepler
    executes it and returns a ready-to-use pandas DataFrame.
    
    Args:
        spl: Custom SPL (Splunk Search Processing Language) query
        index: Splunk index name (required if spl not provided)
        query_type: Type of query - "events", "metrics", "custom", or "auto"
        time_range: Time range (e.g., "-24h", "-7d") - used for auto-generated queries
        earliest: Earliest time for search (e.g., "-30d", "-1h", "@d-7d") 
        latest: Latest time for search (e.g., "now", "@d", "+1d") - defaults to "now"
        limit: Maximum number of results (default: 10000)
        
    Returns:
        pandas.DataFrame: Ready-to-use data for ML
        
    Examples:
        >>> # Custom SPL with automatic time range injection
        >>> data = kp.data.from_splunk(
        ...     spl="search index=sensor_events temperature>50",
        ...     earliest="-30d", latest="now"
        ... )
        
        >>> # Custom SPL with specific time windows
        >>> data = kp.data.from_splunk(
        ...     spl="| mstats avg(_value) WHERE index=metrics metric_name=*",
        ...     earliest="-7d@d", latest="@d"  # Last week, full days
        ... )
        
        >>> # Auto-generate SPL for metrics with time range
        >>> data = kp.data.from_splunk(
        ...     index="sensor_metrics", 
        ...     query_type="metrics",
        ...     earliest="-24h", latest="now"
        ... )
        
        >>> # Business hours only (9 AM to 5 PM today)
        >>> data = kp.data.from_splunk(
        ...     spl="search index=logs error",
        ...     earliest="@d+9h", latest="@d+17h"
        ... )
    """
    
    # Import inside function to avoid circular imports
    from kepler.core.config import load_config
    from kepler.connectors.splunk import SplunkConnector
    
    # Auto-load configuration (transparent to user)
    try:
        config = load_config()
        splunk_config = config.splunk
        
        # Get credentials from environment (transparent)
        token = os.getenv('SPLUNK_TOKEN')
        if not token:
            raise ValueError("❌ SPLUNK_TOKEN not found. Check your .env file.")
            
    except Exception as e:
        raise ValueError(f"❌ Configuration error: {e}")
    
    # Create connection (transparent)
    splunk = SplunkConnector(
        host=splunk_config.host,
        token=token,
        verify_ssl=splunk_config.verify_ssl
    )
    
    # Build SPL query based on user preferences
    if spl is None:
        # Auto-generate SPL based on query_type
        if not index:
            raise ValueError("❌ Either 'spl' or 'index' parameter is required")
            
        if query_type == "custom":
            raise ValueError("❌ query_type='custom' requires 'spl' parameter")
            
        elif query_type == "metrics" or (query_type == "auto" and "metric" in index.lower()):
            # Generate mstats query for metrics indexes - use earliest/latest if provided
            time_clause = _build_time_clause(earliest, latest, time_range)
            spl = f"| mstats avg(_value) as value WHERE index={index}{time_clause} by metric_name span=5m | head {limit}"
            
        elif query_type == "events" or query_type == "auto":
            # Generate search query for event indexes - use earliest/latest if provided
            time_clause = _build_time_clause(earliest, latest, time_range)
            spl = f"search index={index}{time_clause} | head {limit}"
            
        else:
            raise ValueError(f"❌ Invalid query_type: {query_type}. Use 'auto', 'events', 'metrics', or 'custom'")
    
    # Apply time range to custom SPL queries (if earliest/latest provided)
    final_query = _inject_time_range(spl, earliest, latest)
    
    # Execute search and return DataFrame (minimal logging for cleaner notebooks)
    try:
        results = splunk.search(final_query)
        
        if isinstance(results, list):
            if len(results) == 0:
                print(f"⚠️  No data found with query: {final_query}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            print(f"✅ Extracted {len(df):,} records from Splunk")
            return df
            
        else:
            print(f"⚠️  Unexpected result format from Splunk")
            return pd.DataFrame()
            
    except Exception as e:
        # Check if it's a Splunk-specific error
        error_msg = str(e)
        if "Splunk query error:" in error_msg:
            # Extract the actual Splunk error message
            splunk_error = error_msg.split("Splunk query error: ")[1].split("',")[0]
            print(f"❌ Splunk Error: {splunk_error}")
            print(f"🔍 Query: {final_query}")
            print("💡 Tip: Check the SPL syntax according to Splunk documentation")
            return pd.DataFrame()
        else:
            # Generic error handling
            raise RuntimeError(f"❌ Error extracting data from Splunk: {e}")


def list_available_data() -> Dict[str, Any]:
    """
    Show what data is available in Splunk for analysis.
    
    Returns:
        dict: Summary of available indexes and data counts
        
    Example:
        >>> import kepler as kp
        >>> available = kp.data.list_available_data()
        >>> print(available)
    """
    
    # Import inside function to avoid circular imports
    from kepler.core.config import load_config
    from kepler.connectors.splunk import SplunkConnector
    
    try:
        config = load_config()
        token = os.getenv('SPLUNK_TOKEN')
        
        splunk = SplunkConnector(
            host=config.splunk.host,
            token=token,
            verify_ssl=config.splunk.verify_ssl
        )
        
        # Check both event and metrics indexes
        summary = {}
        
        # Events
        try:
            events_validation = splunk.validate_index_access(config.splunk.events_index)
            summary['events'] = {
                'index': config.splunk.events_index,
                'count': events_validation['event_count'],
                'size_mb': events_validation.get('size_mb', 'N/A')
            }
        except:
            summary['events'] = {'index': config.splunk.events_index, 'count': 0}
        
        # Metrics
        try:
            metrics_validation = splunk.validate_index_access(config.splunk.metrics_index)
            summary['metrics'] = {
                'index': config.splunk.metrics_index,
                'count': metrics_validation['event_count'],
                'size_mb': metrics_validation.get('size_mb', 'N/A')
            }
        except:
            summary['metrics'] = {'index': config.splunk.metrics_index, 'count': 0}
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}


def _build_time_clause(earliest: str = None, latest: str = None, time_range: str = "-24h") -> str:
    """
    Build time clause for SPL queries.
    
    Args:
        earliest: Earliest time modifier (e.g., "-30d", "@d-7d")
        latest: Latest time modifier (e.g., "now", "@d")
        time_range: Fallback time range for auto-generated queries
        
    Returns:
        String with time clause for SPL (e.g., " earliest=-30d latest=now")
    """
    if earliest is not None:
        # Use explicit earliest/latest parameters
        latest = latest or "now"  # Default to "now" if not specified
        return f" earliest={earliest} latest={latest}"
    else:
        # Use time_range for backward compatibility
        return f" earliest={time_range}"


def _inject_time_range(spl: str, earliest: str = None, latest: str = None) -> str:
    """
    Inject time range into custom SPL queries if earliest/latest are provided.
    
    Uses a simple, safe approach: add time constraints at the end of the query.
    This works for all Splunk commands and is the most reliable method.
    
    Args:
        spl: Original SPL query
        earliest: Earliest time modifier
        latest: Latest time modifier
        
    Returns:
        Modified SPL query with time constraints
    """
    if earliest is None:
        # No time injection needed
        return spl
    
    latest = latest or "now"
    
    # Simple, safe approach: append time range at the end
    # This works for all SPL commands (search, mstats, mcatalog, etc.)
    
    # Check if query already has time constraints to avoid duplication
    spl_lower = spl.lower()
    if "earliest=" in spl_lower or "latest=" in spl_lower:
        # Query already has time constraints, don't inject
        return spl
    
    # For search commands, ensure proper format
    if not spl.strip().startswith("|") and not spl.lower().strip().startswith("search "):
        # This looks like a raw search, add "search " prefix
        return f"search {spl} earliest={earliest} latest={latest}"
    else:
        # For all other commands (mstats, mcatalog, etc.), append at the end
        return f"{spl} earliest={earliest} latest={latest}"