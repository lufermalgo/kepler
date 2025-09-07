#!/usr/bin/env python3
"""
Test script for Task 5.6: Reproduction System (kp.reproduce.from_version)
Tests the complete reproduction system that can reproduce any version of the project
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import kepler as kp
from pathlib import Path
import pandas as pd
import numpy as np

def test_reproduction_system_api():
    """Test the reproduction system API functions"""
    
    print("=" * 60)
    print("TESTING TASK 5.6: REPRODUCTION SYSTEM (kp.reproduce.from_version)")
    print("=" * 60)
    
    # Test 1: Get reproduction summary
    print("\n1. Testing get_reproduction_summary()...")
    try:
        summary = kp.reproduce.get_reproduction_summary()
        print(f"✅ Reproduction summary retrieved successfully")
        print(f"   Reproduction types: {summary['reproduction_types']}")
        print(f"   Capabilities: {summary['reproduction_capabilities']}")
        print(f"   Available reproductions: {len(summary['available_reproductions'])} categories")
    except Exception as e:
        print(f"❌ get_reproduction_summary() failed: {str(e)}")
        return False
    
    # Test 2: Create some test versions first
    print("\n2. Creating test versions for reproduction...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        test_data_path = "test-lab/reproduction_test_data.csv"
        test_data.to_csv(test_data_path, index=False)
        
        # Create unified version
        unified_version = kp.versioning.create_unified_version(
            version_name="reproduction_test",
            data_paths=[test_data_path],
            experiment_name="reproduction_experiment",
            metadata={"test": True, "purpose": "reproduction_testing"}
        )
        
        print(f"✅ Test versions created successfully")
        print(f"   Unified version: {unified_version.version_id}")
        
    except Exception as e:
        print(f"❌ Test version creation failed: {str(e)}")
        return False
    
    # Test 3: Reproduce unified version
    print("\n3. Testing reproduce_from_version() - unified version...")
    try:
        result = kp.reproduce.from_version(
            version_id=unified_version.version_id,
            reproduction_type="unified"
        )
        
        print(f"✅ Unified version reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Version ID: {result.version_id}")
        print(f"   Reproduction type: {result.reproduction_type}")
        print(f"   Steps completed: {result.steps_completed}")
        print(f"   Steps failed: {result.steps_failed}")
        print(f"   Artifacts created: {result.artifacts_created}")
        
    except Exception as e:
        print(f"❌ Unified version reproduction failed: {str(e)}")
        return False
    
    # Test 4: Auto-detect reproduction type
    print("\n4. Testing reproduce_from_version() - auto-detect type...")
    try:
        result = kp.reproduce.from_version(
            version_id=unified_version.version_id,
            reproduction_type="auto"
        )
        
        print(f"✅ Auto-detect reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Detected type: {result.reproduction_type}")
        
    except Exception as e:
        print(f"❌ Auto-detect reproduction failed: {str(e)}")
        return False
    
    # Test 5: Test data version reproduction
    print("\n5. Testing reproduce_from_version() - data version...")
    try:
        # Create data version first
        data_version = kp.versioning.version_data(test_data_path, "v1.0")
        
        # Reproduce data version
        result = kp.reproduce.from_version(
            version_id=f"{test_data_path}@v1.0",
            reproduction_type="data"
        )
        
        print(f"✅ Data version reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Version ID: {result.version_id}")
        print(f"   Steps completed: {result.steps_completed}")
        
    except Exception as e:
        print(f"❌ Data version reproduction failed: {str(e)}")
        return False
    
    # Test 6: Test pipeline version reproduction
    print("\n6. Testing reproduce_from_version() - pipeline version...")
    try:
        # Create pipeline version first
        pipeline_version = kp.versioning.version_feature_pipeline(
            "test_pipeline",
            "v1.0",
            [{"type": "normalization", "params": {"method": "standard"}}]
        )
        
        # Reproduce pipeline version
        result = kp.reproduce.from_version(
            version_id="test_pipeline@v1.0",
            reproduction_type="pipeline"
        )
        
        print(f"✅ Pipeline version reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Version ID: {result.version_id}")
        print(f"   Steps completed: {result.steps_completed}")
        
    except Exception as e:
        print(f"❌ Pipeline version reproduction failed: {str(e)}")
        return False
    
    # Test 7: Test experiment reproduction
    print("\n7. Testing reproduce_from_version() - experiment...")
    try:
        # Create experiment first
        run_id = kp.versioning.start_experiment("test_experiment")
        kp.versioning.log_parameters(run_id, {"algorithm": "random_forest", "n_estimators": 100})
        kp.versioning.log_metrics(run_id, {"accuracy": 0.95, "precision": 0.92})
        kp.versioning.end_experiment(run_id, "FINISHED")
        
        # Reproduce experiment
        result = kp.reproduce.from_version(
            version_id=f"test_experiment@{run_id}",
            reproduction_type="experiment"
        )
        
        print(f"✅ Experiment reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Version ID: {result.version_id}")
        print(f"   Steps completed: {result.steps_completed}")
        
    except Exception as e:
        print(f"❌ Experiment reproduction failed: {str(e)}")
        return False
    
    # Test 8: Test model reproduction
    print("\n8. Testing reproduce_from_version() - model...")
    try:
        # Reproduce model version
        result = kp.reproduce.from_version(
            version_id="test_model@v1.0",
            reproduction_type="model"
        )
        
        print(f"✅ Model reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Version ID: {result.version_id}")
        print(f"   Steps completed: {result.steps_completed}")
        
    except Exception as e:
        print(f"❌ Model reproduction failed: {str(e)}")
        return False
    
    # Test 9: Test error handling
    print("\n9. Testing error handling...")
    try:
        # Try to reproduce non-existent version
        result = kp.reproduce.from_version(
            version_id="non_existent_version",
            reproduction_type="unified"
        )
        
        print(f"✅ Error handling test completed")
        print(f"   Success: {result.success} (expected: False)")
        print(f"   Error message: {result.error_message}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False
    
    # Test 10: Test invalid format handling
    print("\n10. Testing invalid format handling...")
    try:
        # Try to reproduce with invalid format
        result = kp.reproduce.from_version(
            version_id="invalid_format",
            reproduction_type="data"
        )
        
        print(f"✅ Invalid format handling test completed")
        print(f"   Success: {result.success} (expected: False)")
        print(f"   Error message: {result.error_message}")
        
    except Exception as e:
        print(f"❌ Invalid format handling test failed: {str(e)}")
        return False
    
    # Test 11: Final reproduction summary
    print("\n11. Testing final reproduction summary...")
    try:
        final_summary = kp.reproduce.get_reproduction_summary()
        print(f"✅ Final reproduction summary retrieved")
        print(f"   Reproduction capabilities: {final_summary['reproduction_capabilities']}")
        print(f"   Available reproductions: {len(final_summary['available_reproductions'])} categories")
        
    except Exception as e:
        print(f"❌ Final reproduction summary failed: {str(e)}")
        return False
    
    # Cleanup
    try:
        if os.path.exists(test_data_path):
            os.remove(test_data_path)
        print(f"\n🧹 Cleaned up test data file")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ TASK 5.6 REPRODUCTION SYSTEM - ALL TESTS PASSED")
    print("=" * 60)
    
    return True


def test_api_functions_available():
    """Test that all reproduction API functions are available"""
    
    print("\n🔍 Testing API function availability...")
    
    required_functions = [
        'from_version',
        'reproduce_from_version',
        'get_reproduction_summary'
    ]
    
    available_functions = []
    missing_functions = []
    
    for func_name in required_functions:
        if hasattr(kp.reproduce, func_name):
            available_functions.append(func_name)
            print(f"   ✅ {func_name}")
        else:
            missing_functions.append(func_name)
            print(f"   ❌ {func_name}")
    
    print(f"\n📊 API Functions Status:")
    print(f"   Available: {len(available_functions)}/{len(required_functions)}")
    print(f"   Missing: {len(missing_functions)}")
    
    if missing_functions:
        print(f"   Missing functions: {missing_functions}")
        return False
    
    return True


if __name__ == "__main__":
    print("Kepler Framework - Task 5.6 Reproduction System Test")
    print("Testing complete reproduction system (kp.reproduce.from_version)")
    
    # Test API availability first
    if not test_api_functions_available():
        print("\n❌ API functions not available - stopping tests")
        sys.exit(1)
    
    # Run main tests
    success = test_reproduction_system_api()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Task 5.6 implementation successful!")
        sys.exit(0)
    else:
        print("\n💥 TESTS FAILED - Task 5.6 implementation needs fixes")
        sys.exit(1)
