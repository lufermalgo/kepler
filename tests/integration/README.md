# 🧪 Integration Tests - Test Environment Setup

This document explains how to set up and run integration tests for the Kepler framework with a real Splunk instance.

## 📋 Prerequisites

### Splunk Instance
You need access to a Splunk instance with:
- **REST API access** (default port 8089)
- **HTTP Event Collector (HEC)** enabled (default port 8088)
- **Valid authentication tokens** for both REST API and HEC

### Recommended Setup
- **Splunk Enterprise** (local or remote instance)
- **Development/Test environment** (not production!)
- **SSL certificates** (or disable SSL verification for testing)

## 🔧 Environment Configuration

### Environment Variables

Set these environment variables before running integration tests:

```bash
# Required for REST API tests
export SPLUNK_HOST="https://localhost:8089"
export SPLUNK_TOKEN="your_splunk_auth_token_here"

# Required for HEC tests  
export SPLUNK_HEC_URL="https://localhost:8088/services/collector"
export SPLUNK_HEC_TOKEN="your_hec_token_here"

# Optional (defaults shown)
export SPLUNK_VERIFY_SSL="false"  # Set to true for production
export SPLUNK_TIMEOUT="30"
```

### Getting Splunk Tokens

#### 1. REST API Token
In Splunk Web:
1. Go to **Settings** → **Tokens**
2. Click **New Token**
3. Provide a name (e.g., "Kepler Integration Tests")
4. Set appropriate permissions
5. Copy the generated token

#### 2. HEC Token
In Splunk Web:
1. Go to **Settings** → **Data Inputs**
2. Click **HTTP Event Collector**
3. Click **New Token**
4. Configure:
   - Name: "Kepler HEC Token"
   - Source type: Leave default or set to "kepler:test"
   - Index: "main" (or your preferred test index)
5. Copy the generated token

## 🚀 Running Integration Tests

### Quick Environment Check
```bash
cd tests/integration
python test_splunk_integration.py
```

This will check connectivity and display the status of your test environment.

### Run All Integration Tests
```bash
# From project root
python -m pytest tests/integration/ -v

# With more verbose output
python -m pytest tests/integration/ -v -s

# Run specific test class
python -m pytest tests/integration/test_splunk_integration.py::TestSplunkConnectorIntegration -v
```

### Skip Integration Tests
If you don't have Splunk available, integration tests will be automatically skipped:

```bash
# Tests will show as "SKIPPED" instead of failing
python -m pytest tests/integration/ -v
```

## 🏗️ Test Environment Setup Options

### Option 1: Local Splunk (Recommended for Development)

1. **Download Splunk Enterprise** from splunk.com
2. **Install locally** following Splunk documentation
3. **Configure for testing:**
   ```bash
   # Start Splunk
   sudo /opt/splunk/bin/splunk start
   
   # Access web interface
   open http://localhost:8000
   ```
4. **Enable HEC:**
   - Settings → Data Inputs → HTTP Event Collector
   - Click "Global Settings" and enable HEC
   - Create new HEC token as described above

### Option 2: Docker Splunk

```bash
# Run Splunk in Docker
docker run -d -p 8000:8000 -p 8089:8089 -p 8088:8088 \
  -e SPLUNK_START_ARGS="--accept-license" \
  -e SPLUNK_PASSWORD="changeme123" \
  splunk/splunk:latest

# Wait for startup (2-3 minutes)
# Access at http://localhost:8000 (admin/changeme123)
```

### Option 3: Remote Splunk Instance

If using a remote Splunk instance:

```bash
export SPLUNK_HOST="https://your-splunk-server.com:8089"
export SPLUNK_HEC_URL="https://your-splunk-server.com:8088/services/collector"
# ... set tokens as above
```

## 📊 Test Coverage

The integration tests cover:

### SplunkConnector Tests
- ✅ Basic connectivity and authentication
- ✅ Simple search execution
- ✅ Search with time ranges
- ✅ DataFrame conversion
- ✅ Metrics search functionality
- ✅ Empty result handling
- ✅ Large result set handling

### HecWriter Tests
- ✅ HEC connectivity
- ✅ Single event writing
- ✅ Batch event writing
- ✅ Single metric writing
- ✅ Batch metric writing
- ✅ DataFrame as events
- ✅ DataFrame as metrics

### End-to-End Workflows
- ✅ Extract → Validate → Write workflow
- ✅ Metrics-specific workflow
- ✅ Data quality validation integration

## 🐛 Troubleshooting

### Common Issues

#### "Could not connect to Splunk"
- Check if Splunk is running: `curl -k https://localhost:8089`
- Verify SPLUNK_HOST and SPLUNK_TOKEN
- Check firewall/network connectivity

#### "HEC connection failed"
- Verify HEC is enabled in Splunk
- Check SPLUNK_HEC_URL and SPLUNK_HEC_TOKEN
- Ensure HEC token has correct permissions

#### "Authentication failed"
- Verify token is valid and not expired
- Check token permissions in Splunk
- Try regenerating the token

#### SSL Certificate Errors
```bash
# For testing only - disable SSL verification
export SPLUNK_VERIFY_SSL="false"
```

### Debug Mode
```bash
# Run with verbose logging
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from tests.integration.test_splunk_integration import check_splunk_availability
print('Available:', check_splunk_availability())
"
```

## ⚠️ Important Notes

### For Development
- Integration tests use **real Splunk APIs**
- Tests write actual data to your Splunk instance
- Use a **development/test Splunk environment**
- Clean up test data periodically

### For CI/CD
```bash
# Example GitHub Actions / CI setup
if [[ -n "$SPLUNK_TOKEN" && -n "$SPLUNK_HEC_TOKEN" ]]; then
  python -m pytest tests/integration/ -v
else
  echo "Skipping integration tests - Splunk credentials not available"
fi
```

### Data Cleanup
Test data is written with identifiable sources:
- Source: `kepler_integration_test`
- Sourcetype: `kepler:test*`

To clean up:
```splunk
index=main source="kepler_integration_test" | delete
```

## 🎯 Next Steps

After successful integration test setup:

1. **Run tests regularly** during development
2. **Add new integration tests** for new features
3. **Monitor test data** in your Splunk instance
4. **Configure CI/CD** to run integration tests
5. **Document any custom test scenarios** needed for your environment

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your Splunk instance configuration
3. Review Splunk logs for detailed error messages
4. Ensure all required tokens and permissions are set up correctly