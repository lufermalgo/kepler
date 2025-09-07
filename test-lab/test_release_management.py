#!/usr/bin/env python3
"""
Test script for Task 5.7: Release Management with Multi-Component Versioning
Tests the complete release management system that integrates all versioning components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import kepler as kp
from pathlib import Path
import pandas as pd
import numpy as np

def test_release_management_api():
    """Test the release management API functions"""
    
    print("=" * 60)
    print("TESTING TASK 5.7: RELEASE MANAGEMENT WITH MULTI-COMPONENT VERSIONING")
    print("=" * 60)
    
    # Test 1: Get release summary (should be empty initially)
    print("\n1. Testing get_release_summary()...")
    try:
        summary = kp.versioning.get_release_summary()
        print(f"✅ Release summary retrieved successfully")
        print(f"   Total releases: {summary['total_releases']}")
        print(f"   Status counts: {summary['status_counts']}")
        print(f"   Release components: {summary['release_components']}")
    except Exception as e:
        print(f"❌ get_release_summary() failed: {str(e)}")
        return False
    
    # Test 2: Create test data for release
    print("\n2. Creating test data for release...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        test_data_path = "test-lab/release_test_data.csv"
        test_data.to_csv(test_data_path, index=False)
        
        print(f"✅ Test data created successfully")
        print(f"   Data path: {test_data_path}")
        
    except Exception as e:
        print(f"❌ Test data creation failed: {str(e)}")
        return False
    
    # Test 3: Create release
    print("\n3. Testing create_release()...")
    try:
        release = kp.versioning.create_release(
            release_name="predictive_maintenance",
            version="1.0.0",
            description="Initial release of predictive maintenance model",
            data_paths=[test_data_path],
            experiment_name="pm_experiment_v1",
            metadata={"model_type": "classification", "target_accuracy": 0.95}
        )
        
        print(f"✅ Release created successfully")
        print(f"   Release ID: {release.release_id}")
        print(f"   Release name: {release.release_name}")
        print(f"   Version: {release.version}")
        print(f"   Status: {release.status}")
        print(f"   Git commit: {release.git_commit[:8]}...")
        print(f"   Components: {list(release.components.keys())}")
        
    except Exception as e:
        print(f"❌ create_release() failed: {str(e)}")
        return False
    
    # Test 4: List releases
    print("\n4. Testing list_releases()...")
    try:
        all_releases = kp.versioning.list_releases()
        draft_releases = kp.versioning.list_releases(status="draft")
        
        print(f"✅ List releases successful")
        print(f"   Total releases: {len(all_releases)}")
        print(f"   Draft releases: {len(draft_releases)}")
        
        for i, r in enumerate(all_releases[:3]):  # Show first 3
            print(f"   {i+1}. {r.release_name} v{r.version} ({r.status})")
        
    except Exception as e:
        print(f"❌ list_releases() failed: {str(e)}")
        return False
    
    # Test 5: Get specific release
    print("\n5. Testing get_release()...")
    try:
        retrieved_release = kp.versioning.get_release(release.release_id)
        
        if retrieved_release:
            print(f"✅ Get release successful")
            print(f"   Retrieved: {retrieved_release.release_name} v{retrieved_release.version}")
            print(f"   Status: {retrieved_release.status}")
            print(f"   Metadata: {retrieved_release.metadata}")
        else:
            print(f"❌ Release not found: {release.release_id}")
            return False
        
    except Exception as e:
        print(f"❌ get_release() failed: {str(e)}")
        return False
    
    # Test 6: Update release status
    print("\n6. Testing update_release_status()...")
    try:
        # Update to ready
        success = kp.versioning.update_release_status(release.release_id, "ready")
        
        if success:
            print(f"✅ Release status updated successfully")
            
            # Verify status change
            updated_release = kp.versioning.get_release(release.release_id)
            print(f"   New status: {updated_release.status}")
        else:
            print(f"❌ Release status update failed")
            return False
        
    except Exception as e:
        print(f"❌ update_release_status() failed: {str(e)}")
        return False
    
    # Test 7: Promote release
    print("\n7. Testing promote_release()...")
    try:
        # Promote from ready to released
        success = kp.versioning.promote_release(release.release_id)
        
        if success:
            print(f"✅ Release promoted successfully")
            
            # Verify promotion
            promoted_release = kp.versioning.get_release(release.release_id)
            print(f"   Promoted status: {promoted_release.status}")
        else:
            print(f"❌ Release promotion failed")
            return False
        
    except Exception as e:
        print(f"❌ promote_release() failed: {str(e)}")
        return False
    
    # Test 8: Create another release for testing
    print("\n8. Creating second release for testing...")
    try:
        release2 = kp.versioning.create_release(
            release_name="predictive_maintenance",
            version="1.1.0",
            description="Minor update with improved features",
            data_paths=[test_data_path],
            experiment_name="pm_experiment_v2",
            metadata={"model_type": "classification", "target_accuracy": 0.96, "improvements": ["feature_engineering", "hyperparameter_tuning"]}
        )
        
        print(f"✅ Second release created successfully")
        print(f"   Release ID: {release2.release_id}")
        print(f"   Version: {release2.version}")
        
    except Exception as e:
        print(f"❌ Second release creation failed: {str(e)}")
        return False
    
    # Test 9: Test release reproduction
    print("\n9. Testing reproduce_release()...")
    try:
        result = kp.versioning.reproduce_release(release.release_id)
        
        print(f"✅ Release reproduction completed")
        print(f"   Success: {result.success}")
        print(f"   Version ID: {result.version_id}")
        print(f"   Steps completed: {result.steps_completed}")
        print(f"   Artifacts created: {result.artifacts_created}")
        
        if "release_info" in result.metadata:
            print(f"   Release info: {result.metadata['release_info']}")
        
    except Exception as e:
        print(f"❌ reproduce_release() failed: {str(e)}")
        return False
    
    # Test 10: Test error handling
    print("\n10. Testing error handling...")
    try:
        # Try to get non-existent release
        non_existent = kp.versioning.get_release("non_existent_release")
        
        if non_existent is None:
            print(f"✅ Error handling test passed - non-existent release returns None")
        else:
            print(f"❌ Error handling test failed - should return None")
            return False
        
        # Try to update status with invalid status
        success = kp.versioning.update_release_status(release.release_id, "invalid_status")
        
        if not success:
            print(f"✅ Error handling test passed - invalid status rejected")
        else:
            print(f"❌ Error handling test failed - should reject invalid status")
            return False
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False
    
    # Test 11: Final release summary
    print("\n11. Testing final release summary...")
    try:
        final_summary = kp.versioning.get_release_summary()
        
        print(f"✅ Final release summary retrieved")
        print(f"   Total releases: {final_summary['total_releases']}")
        print(f"   Status counts: {final_summary['status_counts']}")
        print(f"   Latest releases: {len(final_summary['latest_releases'])} categories")
        
        # Show latest releases by status
        for status, info in final_summary['latest_releases'].items():
            print(f"   Latest {status}: {info['release_name']} v{info['version']}")
        
    except Exception as e:
        print(f"❌ Final release summary failed: {str(e)}")
        return False
    
    # Cleanup
    try:
        if os.path.exists(test_data_path):
            os.remove(test_data_path)
        print(f"\n🧹 Cleaned up test data file")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ TASK 5.7 RELEASE MANAGEMENT - ALL TESTS PASSED")
    print("=" * 60)
    
    return True


def test_api_functions_available():
    """Test that all release management API functions are available"""
    
    print("\n🔍 Testing API function availability...")
    
    required_functions = [
        'create_release',
        'list_releases',
        'get_release',
        'update_release_status',
        'promote_release',
        'get_release_summary',
        'reproduce_release'
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
    print("Kepler Framework - Task 5.7 Release Management Test")
    print("Testing complete release management system with multi-component versioning")
    
    # Test API availability first
    if not test_api_functions_available():
        print("\n❌ API functions not available - stopping tests")
        sys.exit(1)
    
    # Run main tests
    success = test_release_management_api()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Task 5.7 implementation successful!")
        sys.exit(0)
    else:
        print("\n💥 TESTS FAILED - Task 5.7 implementation needs fixes")
        sys.exit(1)
