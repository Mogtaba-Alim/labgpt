#!/usr/bin/env python3
"""
test_session.py

Test script to verify session manager functionality
"""

from session_manager import SessionManager
import time

def test_session_manager():
    print("Testing Session Manager...")
    
    # Initialize session manager
    sm = SessionManager()
    
    # Generate a session ID
    session_id = sm.generate_session_id()
    print(f"Generated session ID: {session_id}")
    
    # Create test data (simulating large document data)
    test_data = {
        'grant_overview': 'This is a test grant overview for machine learning research',
        'processed_docs': [
            {
                'filename': 'test_doc.pdf',
                'summary': 'This is a test document summary',
                'document_type': 'grant',
                'word_count': 1000,
                'sections': {
                    'Background': 'Test background content ' * 100,  # Large content
                    'Methods': 'Test methods content ' * 100,
                    'Results': 'Test results content ' * 100,
                }
            }
        ],
        'completed_sections': {
            'Background': 'This is the completed background section',
            'Methods': 'This is the completed methods section'
        }
    }
    
    # Save the data
    success = sm.save_session_data(session_id, test_data)
    print(f"Save operation successful: {success}")
    
    # Load the data back
    loaded_data = sm.load_session_data(session_id)
    print(f"Load operation successful: {loaded_data is not None}")
    
    if loaded_data:
        print(f"Grant overview matches: {loaded_data['grant_overview'] == test_data['grant_overview']}")
        print(f"Number of processed docs: {len(loaded_data['processed_docs'])}")
        print(f"Number of completed sections: {len(loaded_data['completed_sections'])}")
    
    # Test update functionality
    updates = {'current_section': 'Methods', 'test_field': 'test_value'}
    success = sm.update_session_data(session_id, updates)
    print(f"Update operation successful: {success}")
    
    # Load updated data
    updated_data = sm.load_session_data(session_id)
    if updated_data:
        print(f"Update applied correctly: {'test_field' in updated_data and updated_data['test_field'] == 'test_value'}")
    
    # Get session info
    info = sm.get_session_info(session_id)
    if info:
        print(f"Session file size: {info['file_size']} bytes")
        print(f"Session created: {info['created']}")
    
    # Clean up
    sm.delete_session(session_id)
    print("Test completed - session cleaned up")

if __name__ == '__main__':
    test_session_manager() 