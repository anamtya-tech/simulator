#!/usr/bin/env python3
"""
Test script to verify classifier output integration with analyzer.

This script:
1. Checks if classifier output files exist
2. Parses them using the new parser
3. Verifies the structure includes YAMNet data
"""

import sys
import json
from pathlib import Path

# Add simulator directory to path
sys.path.insert(0, '/home/azureuser/simulator')

from odas_classifier_parser import OdasClassifierParser

def test_parser():
    """Test the ODAS classifier parser"""
    print("=" * 60)
    print("Testing ODAS Classifier Output Parser")
    print("=" * 60)
    
    parser = OdasClassifierParser()
    
    # Find latest session file
    latest_file = parser.get_latest_session_file()
    
    if not latest_file:
        print("\n❌ No session files found in /home/azureuser/sodas/ClassifierLogs/")
        print("\nTo generate test data:")
        print("1. Open the Streamlit app: streamlit run /home/azureuser/simulator/app.py")
        print("2. Go to '⚙️ ODAS Simulator'")
        print("3. Run a simulation")
        print("4. Check that 'enable_classifier_output = \"enabled\"' in /home/azureuser/sodas/local_socket.cfg")
        return False
    
    print(f"\n✅ Found session file: {latest_file.name}")
    print(f"   Path: {latest_file}")
    
    # Parse frames
    try:
        frames = parser.parse_session_file(str(latest_file))
        print(f"\n✅ Parsed {len(frames)} frames")
    except Exception as e:
        print(f"\n❌ Error parsing file: {e}")
        return False
    
    if not frames:
        print("\n❌ No frames found in file")
        return False
    
    # Check first frame structure
    first_frame = frames[0]
    print(f"\n✅ First frame timestamp: {first_frame.get('timeStamp')}")
    
    tracks = parser.extract_tracks(first_frame)
    print(f"✅ Tracks in first frame: {len(tracks)}")
    
    if tracks:
        print("\n📊 Sample Track Data:")
        track = tracks[0]
        print(f"   Track ID: {track.id}")
        print(f"   Tag: {track.tag}")
        print(f"   Direction: ({track.x:.3f}, {track.y:.3f}, {track.z:.3f})")
        print(f"   Activity: {track.activity:.3f}")
        print(f"   Type: {track.type}")
        print(f"   Frame Count: {track.frame_count}")
        print(f"   YAMNet Class ID: {track.class_id}")
        print(f"   YAMNet Class Name: {track.class_name}")
        print(f"   YAMNet Confidence: {track.class_confidence:.3f}")
        print(f"   Classification Timestamp: {track.class_timestamp}")
        
        # Verify required fields exist
        required_fields = ['id', 'x', 'y', 'z', 'class_name', 'class_confidence']
        missing = [f for f in required_fields if getattr(track, f, None) is None]
        if missing:
            print(f"\n⚠️  Missing fields: {missing}")
        else:
            print("\n✅ All required fields present")
    
    # Generate summary
    print("\n📈 Session Summary:")
    summary = parser.summarize_session(str(latest_file))
    print(f"   Total frames: {summary['total_frames']}")
    print(f"   Total track detections: {summary['total_track_detections']}")
    print(f"   Unique tracks: {summary['unique_tracks']}")
    
    if summary['class_distribution']:
        print(f"\n🎯 Classification Distribution:")
        for class_name, count in sorted(summary['class_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"      {class_name}: {count}")
    else:
        print(f"\n⚠️  No classified tracks found")
        print(f"   This might mean:")
        print(f"   - No tracks accumulated 96 frames yet")
        print(f"   - YAMNet classification is not running")
    
    return True


def test_analyzer_integration():
    """Test that analyzer can read the new format"""
    print("\n" + "=" * 60)
    print("Testing Analyzer Integration")
    print("=" * 60)
    
    # Check if there are any run files
    runs_dir = Path("/home/azureuser/simulator/outputs/runs")
    if not runs_dir.exists():
        print("\n❌ Runs directory not found")
        return False
    
    run_files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        print("\n❌ No run files found")
        print("\nPlease run a simulation first using the Streamlit app")
        return False
    
    latest_run = run_files[0]
    print(f"\n✅ Found run file: {latest_run.name}")
    
    # Load run data
    with open(latest_run, 'r') as f:
        run_data = json.load(f)
    
    # Check for required fields
    print(f"\n📋 Run Data Structure:")
    print(f"   Run ID: {run_data.get('run_id', 'N/A')}")
    print(f"   Scene: {run_data.get('scene_name', 'N/A')}")
    print(f"   Session Live File: {run_data.get('session_live_file', 'N/A')}")
    
    session_live_file = run_data.get('session_live_file')
    if session_live_file and Path(session_live_file).exists():
        print(f"\n✅ Session live file exists and is accessible")
        
        # Quick check of content
        with open(session_live_file, 'r') as f:
            first_line = f.readline()
            if first_line.strip():
                data = json.loads(first_line)
                if 'src' in data and len(data['src']) > 0:
                    src = data['src'][0]
                    has_classification = 'class_name' in src and 'class_confidence' in src
                    if has_classification:
                        print(f"✅ Session file contains classification data")
                        print(f"   Sample: class_name={src.get('class_name')}, "
                              f"confidence={src.get('class_confidence', 0):.3f}")
                    else:
                        print(f"⚠️  Session file missing classification fields")
                        print(f"   Available fields: {list(src.keys())}")
    else:
        print(f"\n❌ Session live file not found or not accessible")
    
    return True


if __name__ == "__main__":
    print("\n🧪 ODAS Classifier Output Test Suite\n")
    
    parser_ok = test_parser()
    analyzer_ok = test_analyzer_integration()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Parser Test: {'✅ PASS' if parser_ok else '❌ FAIL'}")
    print(f"Analyzer Integration: {'✅ PASS' if analyzer_ok else '❌ FAIL'}")
    print()
    
    if parser_ok and analyzer_ok:
        print("✅ All tests passed!")
        print("\n📝 Next steps:")
        print("1. Open Streamlit app: streamlit run /home/azureuser/simulator/app.py")
        print("2. Run a simulation in '⚙️ ODAS Simulator'")
        print("3. View results in '📊 Results Analyzer'")
        print("4. Check for YAMNet classification statistics")
    else:
        print("⚠️  Some tests failed. See output above for details.")
