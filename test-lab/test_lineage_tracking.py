#!/usr/bin/env python3
"""
Test script for Task 5.5: End-to-End Traceability and Lineage Tracking
Tests the complete lineage tracking system that connects data, pipelines, experiments, and models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import kepler as kp
from pathlib import Path
import pandas as pd
import numpy as np

def test_lineage_tracking_api():
    """Test the lineage tracking API functions"""
    
    print("=" * 60)
    print("TESTING TASK 5.5: END-TO-END TRACEABILITY AND LINEAGE TRACKING")
    print("=" * 60)
    
    # Test 1: Create data lineage nodes
    print("\n1. Testing create_data_lineage()...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        test_data_path = "test-lab/sensor_data.csv"
        test_data.to_csv(test_data_path, index=False)
        
        # Create data lineage node
        data_node = kp.versioning.create_data_lineage(
            data_path=test_data_path,
            data_version="v1.0",
            metadata={"source": "sensors", "rows": 100, "columns": 3}
        )
        
        print(f"✅ Data lineage node created successfully")
        print(f"   Node ID: {data_node.node_id}")
        print(f"   Node type: {data_node.node_type}")
        print(f"   Name: {data_node.name}")
        print(f"   Version: {data_node.version}")
        
    except Exception as e:
        print(f"❌ create_data_lineage() failed: {str(e)}")
        return False
    
    # Test 2: Create pipeline lineage node
    print("\n2. Testing create_pipeline_lineage()...")
    try:
        pipeline_node = kp.versioning.create_pipeline_lineage(
            pipeline_name="feature_engineering",
            pipeline_version="v1.0",
            input_data_nodes=[data_node.node_id],
            metadata={"steps": ["normalization", "encoding", "selection"], "framework": "sklearn"}
        )
        
        print(f"✅ Pipeline lineage node created successfully")
        print(f"   Node ID: {pipeline_node.node_id}")
        print(f"   Input nodes: {pipeline_node.inputs}")
        print(f"   Metadata: {pipeline_node.metadata}")
        
    except Exception as e:
        print(f"❌ create_pipeline_lineage() failed: {str(e)}")
        return False
    
    # Test 3: Create experiment lineage node
    print("\n3. Testing create_experiment_lineage()...")
    try:
        experiment_node = kp.versioning.create_experiment_lineage(
            experiment_name="model_training",
            run_id="run_001",
            input_pipeline_nodes=[pipeline_node.node_id],
            metadata={"algorithm": "random_forest", "hyperparameters": {"n_estimators": 100}}
        )
        
        print(f"✅ Experiment lineage node created successfully")
        print(f"   Node ID: {experiment_node.node_id}")
        print(f"   Input nodes: {experiment_node.inputs}")
        print(f"   Metadata: {experiment_node.metadata}")
        
    except Exception as e:
        print(f"❌ create_experiment_lineage() failed: {str(e)}")
        return False
    
    # Test 4: Create model lineage node
    print("\n4. Testing create_model_lineage()...")
    try:
        model_node = kp.versioning.create_model_lineage(
            model_name="predictive_maintenance",
            model_version="v1.0",
            input_experiment_nodes=[experiment_node.node_id],
            metadata={"accuracy": 0.95, "precision": 0.92, "recall": 0.88, "framework": "sklearn"}
        )
        
        print(f"✅ Model lineage node created successfully")
        print(f"   Node ID: {model_node.node_id}")
        print(f"   Input nodes: {model_node.inputs}")
        print(f"   Metadata: {model_node.metadata}")
        
    except Exception as e:
        print(f"❌ create_model_lineage() failed: {str(e)}")
        return False
    
    # Test 5: Create deployment lineage node
    print("\n5. Testing create_deployment_lineage()...")
    try:
        deployment_node = kp.versioning.create_deployment_lineage(
            deployment_name="production_api",
            deployment_version="v1.0",
            input_model_nodes=[model_node.node_id],
            metadata={"endpoint": "https://api.example.com/predict", "environment": "production"}
        )
        
        print(f"✅ Deployment lineage node created successfully")
        print(f"   Node ID: {deployment_node.node_id}")
        print(f"   Input nodes: {deployment_node.inputs}")
        print(f"   Metadata: {deployment_node.metadata}")
        
    except Exception as e:
        print(f"❌ create_deployment_lineage() failed: {str(e)}")
        return False
    
    # Test 6: Get lineage graph
    print("\n6. Testing get_lineage_graph()...")
    try:
        lineage_graph = kp.versioning.get_lineage_graph()
        
        print(f"✅ Lineage graph retrieved successfully")
        print(f"   Total nodes: {len(lineage_graph['nodes'])}")
        print(f"   Total edges: {len(lineage_graph['edges'])}")
        
        # Show node types
        node_types = {}
        for node in lineage_graph['nodes'].values():
            node_type = node['node_type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        print(f"   Node types: {node_types}")
        
    except Exception as e:
        print(f"❌ get_lineage_graph() failed: {str(e)}")
        return False
    
    # Test 7: Get node lineage
    print("\n7. Testing get_node_lineage()...")
    try:
        # Test upstream lineage
        upstream_lineage = kp.versioning.get_node_lineage(model_node.node_id, "upstream")
        print(f"✅ Upstream lineage retrieved successfully")
        print(f"   Node: {upstream_lineage['node']['name']}")
        print(f"   Upstream nodes: {len(upstream_lineage['upstream'])}")
        
        # Test downstream lineage
        downstream_lineage = kp.versioning.get_node_lineage(model_node.node_id, "downstream")
        print(f"✅ Downstream lineage retrieved successfully")
        print(f"   Downstream nodes: {len(downstream_lineage['downstream'])}")
        
        # Test both directions
        both_lineage = kp.versioning.get_node_lineage(model_node.node_id, "both")
        print(f"✅ Both directions lineage retrieved successfully")
        print(f"   Total connected nodes: {len(both_lineage['upstream']) + len(both_lineage['downstream'])}")
        
    except Exception as e:
        print(f"❌ get_node_lineage() failed: {str(e)}")
        return False
    
    # Test 8: Get complete lineage
    print("\n8. Testing get_complete_lineage()...")
    try:
        complete_lineage = kp.versioning.get_complete_lineage()
        
        print(f"✅ Complete lineage retrieved successfully")
        print(f"   Total nodes: {complete_lineage['total_nodes']}")
        print(f"   Total edges: {complete_lineage['total_edges']}")
        print(f"   Node counts: {complete_lineage['node_counts']}")
        print(f"   Edge counts: {complete_lineage['edge_counts']}")
        print(f"   Root nodes: {len(complete_lineage['root_nodes'])}")
        print(f"   Leaf nodes: {len(complete_lineage['leaf_nodes'])}")
        print(f"   Lineage completeness: {complete_lineage['lineage_completeness']}")
        
    except Exception as e:
        print(f"❌ get_complete_lineage() failed: {str(e)}")
        return False
    
    # Test 9: Trace data flow
    print("\n9. Testing trace_data_flow()...")
    try:
        # Trace from data to deployment
        flow_path = kp.versioning.trace_data_flow(data_node.node_id, deployment_node.node_id)
        
        print(f"✅ Data flow traced successfully")
        print(f"   Path length: {len(flow_path)}")
        print(f"   Path: {' -> '.join(flow_path)}")
        
        # Test reverse flow (should be empty)
        reverse_flow = kp.versioning.trace_data_flow(deployment_node.node_id, data_node.node_id)
        print(f"   Reverse flow: {len(reverse_flow)} (expected: 0)")
        
    except Exception as e:
        print(f"❌ trace_data_flow() failed: {str(e)}")
        return False
    
    # Test 10: Create additional nodes for complex lineage
    print("\n10. Testing complex lineage scenario...")
    try:
        # Create another data node
        data_node2 = kp.versioning.create_data_lineage(
            data_path="test-lab/weather_data.csv",
            data_version="v1.0",
            metadata={"source": "weather_api", "rows": 50, "columns": 5}
        )
        
        # Create a pipeline that uses both data sources
        pipeline_node2 = kp.versioning.create_pipeline_lineage(
            pipeline_name="data_fusion",
            pipeline_version="v1.0",
            input_data_nodes=[data_node.node_id, data_node2.node_id],
            metadata={"operation": "merge", "join_key": "timestamp"}
        )
        
        print(f"✅ Complex lineage scenario created successfully")
        print(f"   Data nodes: 2")
        print(f"   Pipeline with multiple inputs: {pipeline_node2.node_id}")
        
    except Exception as e:
        print(f"❌ Complex lineage scenario failed: {str(e)}")
        return False
    
    # Cleanup
    try:
        if os.path.exists(test_data_path):
            os.remove(test_data_path)
        print(f"\n🧹 Cleaned up test data file")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ TASK 5.5 END-TO-END TRACEABILITY - ALL TESTS PASSED")
    print("=" * 60)
    
    return True


def test_api_functions_available():
    """Test that all lineage tracking API functions are available"""
    
    print("\n🔍 Testing API function availability...")
    
    required_functions = [
        'create_data_lineage',
        'create_pipeline_lineage',
        'create_experiment_lineage',
        'create_model_lineage',
        'create_deployment_lineage',
        'get_lineage_graph',
        'get_node_lineage',
        'get_complete_lineage',
        'trace_data_flow'
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
    print("Kepler Framework - Task 5.5 End-to-End Traceability Test")
    print("Testing complete lineage tracking system")
    
    # Test API availability first
    if not test_api_functions_available():
        print("\n❌ API functions not available - stopping tests")
        sys.exit(1)
    
    # Run main tests
    success = test_lineage_tracking_api()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Task 5.5 implementation successful!")
        sys.exit(0)
    else:
        print("\n💥 TESTS FAILED - Task 5.5 implementation needs fixes")
        sys.exit(1)
