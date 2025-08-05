# Professional Data Analysis Notebooks

## Overview
This directory contains professional data analysis examples using the Kepler framework. These notebooks demonstrate real-world industrial sensor data analysis for team presentations and stakeholder reviews.

## Available Notebooks

### 1. events_analysis.py
**Purpose**: Comprehensive analysis of sensor event data from Splunk's event index

**Key Features**:
- Data extraction using custom SPL queries
- Data quality assessment and preprocessing
- Exploratory data analysis with statistical summaries
- Professional visualizations (distribution charts, time series, correlation analysis)
- Anomaly detection using statistical methods
- Machine learning data preparation
- Feature engineering and export

**Target Audience**: Data scientists, analysts, and engineers working with detailed sensor event data

**Output**: Processed dataset ready for ML applications

### 2. metrics_analysis.py
**Purpose**: Time-series analysis of aggregated sensor metrics data

**Key Features**:
- Metrics data extraction using mstats queries
- Time-series trend analysis and pattern recognition
- Statistical correlation analysis between metrics
- Advanced forecasting with moving averages
- Peak detection and anomaly identification
- Professional time-series visualizations
- Forecast accuracy assessment

**Target Audience**: Operations teams, data engineers, and analysts focused on monitoring and forecasting

**Output**: Time-series dataset with statistical features

## Usage Instructions

### Prerequisites
1. Ensure Kepler framework is properly installed
2. Configure Splunk connectivity (kepler.yml and .env)
3. Verify data availability in kepler_lab and kepler_metrics indexes

### Running the Analysis

#### Option 1: As Python Scripts
```bash
cd test-lab/notebooks
python events_analysis.py
python metrics_analysis.py
```

#### Option 2: Convert to Jupyter Notebooks
```bash
# Install jupytext for conversion
pip install jupytext

# Convert to notebooks
jupytext --to notebook events_analysis.py
jupytext --to notebook metrics_analysis.py

# Launch Jupyter
jupyter lab
```

#### Option 3: Use in IDE
- Open files directly in your preferred IDE (VSCode, Cursor AI, PyCharm)
- Execute code sections interactively
- Modify analysis parameters as needed

## Code Standards

### Professional Coding Practices
- No emojis or casual language
- Clean, readable code with proper documentation
- Statistical rigor in analysis methods
- Professional visualization standards
- Comprehensive error handling

### Output Standards
- Clear section headers and documentation
- Statistical summaries with proper formatting
- Professional-grade visualizations
- Actionable insights and recommendations
- Exportable datasets for further analysis

## Customization

### Modifying Analysis Parameters
- **Time Range**: Update `earliest=-24h` in SPL queries
- **Data Volume**: Adjust `head 1000` limits in extraction queries
- **Visualization**: Modify plot settings in matplotlib/seaborn sections
- **Statistical Methods**: Update anomaly detection thresholds and methods

### Adding New Analysis
- Follow existing code structure and documentation standards
- Maintain professional tone and clear commenting
- Include statistical validation of results
- Export processed data for reproducibility

## Team Presentation Guidelines

### For Technical Teams
- Focus on methodology and statistical rigor
- Emphasize data quality and preprocessing steps
- Highlight anomaly detection and forecasting capabilities
- Discuss feature engineering and ML preparation

### For Business Stakeholders
- Emphasize operational insights and recommendations
- Focus on visualization of key trends and patterns
- Highlight actionable findings for operations improvement
- Discuss ROI potential of predictive analytics

### For Management
- Summary of key findings and business impact
- Risk identification and mitigation strategies
- Resource requirements for implementation
- Timeline for deploying insights into operations

## Next Steps

1. **Model Development**: Use prepared datasets for machine learning model training
2. **Dashboard Creation**: Convert visualizations into operational dashboards
3. **Automation**: Implement automated analysis pipelines
4. **Integration**: Connect insights to operational workflows
5. **Monitoring**: Establish baseline metrics and performance monitoring

## Support and Documentation

- **Framework Documentation**: See main README.md in project root
- **Splunk Queries**: Refer to Splunk SPL documentation for query customization
- **Statistical Methods**: Consult data science team for advanced statistical analysis
- **Operational Integration**: Work with operations team for deployment planning