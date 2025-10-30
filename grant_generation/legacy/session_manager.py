#!/usr/bin/env python3
"""
session_manager.py

Session management for large data that exceeds cookie limits.
Stores session data in files and only keeps session IDs in cookies.
"""

import os
import json
import pickle
import uuid
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

class SessionManager:
    def __init__(self, storage_dir: str = "session_storage", session_timeout_hours: int = 24):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # Clean up old sessions on initialization
        self.cleanup_expired_sessions()
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session"""
        return self.storage_dir / f"session_{session_id}.pkl"
    
    def save_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Save session data to file"""
        try:
            session_file = self.get_session_file_path(session_id)
            
            # Add timestamp
            data['_timestamp'] = datetime.now().isoformat()
            data['_session_id'] = session_id
            
            with open(session_file, 'wb') as f:
                pickle.dump(data, f)
            
            logging.info(f"Saved session data for {session_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving session {session_id}: {e}")
            return False
    
    def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from file"""
        try:
            session_file = self.get_session_file_path(session_id)
            
            if not session_file.exists():
                logging.warning(f"Session file not found: {session_id}")
                return None
            
            with open(session_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if session is expired
            if '_timestamp' in data:
                timestamp = datetime.fromisoformat(data['_timestamp'])
                if datetime.now() - timestamp > self.session_timeout:
                    logging.info(f"Session {session_id} expired, removing")
                    self.delete_session(session_id)
                    return None
            
            logging.info(f"Loaded session data for {session_id}")
            return data
            
        except Exception as e:
            logging.error(f"Error loading session {session_id}: {e}")
            return None
    
    def update_session_data(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields in session data"""
        try:
            # Load existing data
            existing_data = self.load_session_data(session_id) or {}
            
            # Update with new data
            existing_data.update(updates)
            
            # Save back
            return self.save_session_data(session_id, existing_data)
            
        except Exception as e:
            logging.error(f"Error updating session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session file"""
        try:
            session_file = self.get_session_file_path(session_id)
            if session_file.exists():
                session_file.unlink()
                logging.info(f"Deleted session {session_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired session files"""
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            for session_file in self.storage_dir.glob("session_*.pkl"):
                try:
                    # Check file modification time
                    file_time = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if current_time - file_time > self.session_timeout:
                        session_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    logging.error(f"Error checking session file {session_file}: {e}")
            
            if cleaned_count > 0:
                logging.info(f"Cleaned up {cleaned_count} expired sessions")
            
            return cleaned_count
            
        except Exception as e:
            logging.error(f"Error during session cleanup: {e}")
            return 0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get basic info about a session without loading full data"""
        try:
            session_file = self.get_session_file_path(session_id)
            if not session_file.exists():
                return None
            
            stat = session_file.stat()
            return {
                'session_id': session_id,
                'file_size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'exists': True
            }
            
        except Exception as e:
            logging.error(f"Error getting session info {session_id}: {e}")
            return None

# Global session manager instance
session_manager = SessionManager() 