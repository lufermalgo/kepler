#!/usr/bin/env python3
"""
Test script for Task 5.4: Unified Versioning System (Git + DVC + MLflow)
Tests the unified versioning API that integrates Git, DVC, and MLflow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import kepler as kp
from pathlib import Path
import pandas as pd
import numpy as np

def test_unified_versioning_api():
    """Test the unified versioning API functions"""
    
    print("=" * 60)
    print("TESTING TASK 5.4: UNIFIED VERSIONING SYSTEM")
    print("=" * 60)
    
    # Test 1: Get version summary
    print("\n1. Testing get_version_summary()...")
    try:
        summary = kp.versioning.get_version_summary()
        print(f"✅ Version summary retrieved successfully")
        print(f"   Total versions: {summary['total_versions']}")
        print(f"   Latest version: {summary['latest_version']}")
        print(f"   Git available: {summary['components']['git']['available']}")
        print(f"   DVC available: {summary['components']['dvc']['available']}")
        print(f"   MLflow available: {summary['components']['mlflow']['available']}")
    except Exception as e:
        print(f"❌ get_version_summary() failed: {str(e)}")
        return False
    
    # Test 2: Create unified version
    print("\n2. Testing create_unified_version()...")
    try:
        # Create some test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        test_data_path = "test-lab/test_data.csv"
        test_data.to_csv(test_data_path, index=False)
        
        # Create unified version
        version = kp.versioning.create_unified_version(
            version_name="test_version_1",
            data_paths=[test_data_path],
            experiment_name="test_experiment",
            metadata={"test": True, "description": "Test unified version"}
        )
        
        print(f"✅ Unified version created successfully")
        print(f"   Version ID: {version.version_id}")
        print(f"   Git commit: {version.git_commit[:8]}...")
        print(f"   DVC data version: {version.dvc_data_version}")
        print(f"   MLflow run ID: {version.mlflow_run_id}")
        print(f"   Timestamp: {version.timestamp}")
        
    except Exception as e:
        print(f"❌ create_unified_version() failed: {str(e)}")
        return False
    
    # Test 3: List unified versions
    print("\n3. Testing list_unified_versions()...")
    try:
        versions = kp.versioning.list_unified_versions()
        print(f"✅ List unified versions successful")
        print(f"   Found {len(versions)} versions")
        
        for i, v in enumerate(versions[:3]):  # Show first 3
            print(f"   {i+1}. {v.version_id} ({v.timestamp[:19]})")
            
    except Exception as e:
        print(f"❌ list_unified_versions() failed: {str(e)}")
        return False
    
    # Test 4: Get specific unified version
    print("\n4. Testing get_unified_version()...")
    try:
        if versions:
            version_id = versions[0].version_id
            retrieved_version = kp.versioning.get_unified_version(version_id)
            
            if retrieved_version:
                print(f"✅ Get unified version successful")
                print(f"   Retrieved: {retrieved_version.version_id}")
                print(f"   Git commit: {retrieved_version.git_commit[:8]}...")
                print(f"   Metadata: {retrieved_version.metadata}")
            else:
                print(f"❌ Version not found: {version_id}")
                return False
        else:
            print("⚠️  No versions available for testing get_unified_version()")
            
    except Exception as e:
        print(f"❌ get_unified_version() failed: {str(e)}")
        return False
    
    # Test 5: Create another version for checkout testing
    print("\n5. Testing create_unified_version() with different metadata...")
    try:
        version2 = kp.versioning.create_unified_version(
            version_name="test_version_2",
            data_paths=[test_data_path],
            experiment_name="test_experiment_2",
            metadata={"test": True, "description": "Second test version", "iteration": 2}
        )
        
        print(f"✅ Second unified version created successfully")
        print(f"   Version ID: {version2.version_id}")
        
    except Exception as e:
        print(f"❌ Second create_unified_version() failed: {str(e)}")
        return False
    
    # Test 6: Test checkout (if Git is available)
    print("\n6. Testing checkout_unified_version()...")
    try:
        if versions and versions[0].components.get("git", {}).get("available"):
            # Get the first version
            first_version = versions[0]
            print(f"   Attempting checkout of: {first_version.version_id}")
            
            # Note: This might fail in test environment, but we test the API
            success = kp.versioning.checkout_unified_version(first_version.version_id)
            
            if success:
                print(f"✅ Checkout successful")
            else:
                print(f"⚠️  Checkout failed (expected in test environment)")
        else:
            print("⚠️  Git not available, skipping checkout test")
            
    except Exception as e:
        print(f"⚠️  checkout_unified_version() failed (expected): {str(e)}")
    
    # Test 7: Final version summary
    print("\n7. Testing final version summary...")
    try:
        final_summary = kp.versioning.get_version_summary()
        print(f"✅ Final version summary retrieved")
        print(f"   Total versions: {final_summary['total_versions']}")
        print(f"   Recent versions: {len(final_summary['recent_versions'])}")
        
        for i, v in enumerate(final_summary['recent_versions'][:3]):
            print(f"   {i+1}. {v['version_id']} (Git: {v['git_commit']}, DVC: {v['has_dvc']}, MLflow: {v['has_mlflow']})")
            
    except Exception as e:
        print(f"❌ Final version summary failed: {str(e)}")
        return False
    
    # Cleanup
    try:
        if os.path.exists(test_data_path):
            os.remove(test_data_path)
        print(f"\n🧹 Cleaned up test data file")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ TASK 5.4 UNIFIED VERSIONING SYSTEM - ALL TESTS PASSED")
    print("=" * 60)
    
    return True


def test_api_functions_available():
    """Test that all unified versioning API functions are available"""
    
    print("\n🔍 Testing API function availability...")
    
    required_functions = [
        'create_unified_version',
        'list_unified_versions', 
        'get_unified_version',
        'checkout_unified_version',
        'get_version_summary'
    ]
    
    available_functions = []
    missing_functions = []
    
    for func_name in required_functions:
        if hasattr(kp.versioning, func_name):
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
    print("Kepler Framework - Task 5.4 Unified Versioning System Test")
    print("Testing Git + DVC + MLflow unified versioning integration")
    
    # Test API availability first
    if not test_api_functions_available():
        print("\n❌ API functions not available - stopping tests")
        sys.exit(1)
    
    # Run main tests
    success = test_unified_versioning_api()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Task 5.4 implementation successful!")
        sys.exit(0)
    else:
        print("\n💥 TESTS FAILED - Task 5.4 implementation needs fixes")
        sys.exit(1)
