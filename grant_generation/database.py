"""
database.py

SQLite database schema and operations for grant generation projects.
Replaces file-based pickle sessions for better scalability and querying.
"""

import sqlite3
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class GrantDatabase:
    """
    SQLite database for grant generation project management.

    Features:
    - Project management (title, overview, index directory)
    - Document tracking (uploaded files, indexing status)
    - Section draft versioning (with parent lineage)
    - Refinement feedback history
    - Generation telemetry (performance metrics)
    """

    def __init__(self, db_path: str = "grant_generation/grants.db"):
        """
        Initialize database connection and create tables.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = db_path

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect with check_same_thread=False for Flask compatibility
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._create_tables()

        logger.info(f"Database initialized: {db_path}")

    def _create_tables(self):
        """Create all database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                overview TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                index_dir TEXT NOT NULL,
                has_base_corpus BOOLEAN DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)

        # Project documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_documents (
                doc_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                doc_type TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed BOOLEAN DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            )
        """)

        # Section drafts table (with versioning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS section_drafts (
                draft_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                section_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                content TEXT NOT NULL,
                citations TEXT,
                quality_score REAL,
                quality_feedback TEXT,
                generation_params TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parent_draft_id TEXT,
                conversation_history TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_draft_id) REFERENCES section_drafts(draft_id),
                UNIQUE(project_id, section_name, version)
            )
        """)

        # Refinement feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refinement_feedback (
                feedback_id TEXT PRIMARY KEY,
                draft_id TEXT NOT NULL,
                feedback_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (draft_id) REFERENCES section_drafts(draft_id) ON DELETE CASCADE
            )
        """)

        # Generation telemetry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_log (
                log_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                section_name TEXT NOT NULL,
                query TEXT,
                num_chunks_retrieved INTEGER,
                cache_hits INTEGER DEFAULT 0,
                cache_misses INTEGER DEFAULT 0,
                generation_time_sec REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            )
        """)

        self.conn.commit()
        logger.debug("Database tables created/verified")

    # ========== PROJECT OPERATIONS ==========

    def create_project(self, title: str, overview: str, index_dir: str, has_base: bool = False) -> str:
        """
        Create a new grant project.

        Args:
            title: Project title
            overview: Project overview/description
            index_dir: Path to RAG index directory
            has_base: Whether base corpus was copied

        Returns:
            project_id: Unique project identifier (UUID)
        """
        project_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO projects (project_id, title, overview, index_dir, has_base_corpus)
            VALUES (?, ?, ?, ?, ?)
        """, (project_id, title, overview, index_dir, int(has_base)))

        self.conn.commit()
        logger.info(f"Created project {project_id}: {title}")

        return project_id

    def get_project(self, project_id: str) -> Optional[Dict]:
        """
        Get project information.

        Args:
            project_id: Project identifier

        Returns:
            Project dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
        row = cursor.fetchone()

        return dict(row) if row else None

    def get_project_field(self, project_id: str, field: str) -> Any:
        """
        Get a specific field from a project.

        Args:
            project_id: Project identifier
            field: Field name (e.g., 'index_dir', 'title')

        Returns:
            Field value or None if project not found
        """
        project = self.get_project(project_id)
        return project.get(field) if project else None

    def update_project(self, project_id: str, **updates) -> bool:
        """
        Update project fields.

        Args:
            project_id: Project identifier
            **updates: Field-value pairs to update

        Returns:
            True if successful, False otherwise
        """
        if not updates:
            return False

        cursor = self.conn.cursor()

        # Build SET clause
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [project_id]

        # Always update updated_at
        query = f"UPDATE projects SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE project_id = ?"

        try:
            cursor.execute(query, values)
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating project {project_id}: {e}")
            return False

    # ========== DOCUMENT OPERATIONS ==========

    def add_document(
        self,
        project_id: str,
        filename: str,
        file_path: str,
        doc_type: str,
        indexed: bool = False,
        chunk_count: int = 0
    ) -> str:
        """
        Add a document to a project.

        Args:
            project_id: Project identifier
            filename: Original filename
            file_path: Path to saved file
            doc_type: Document type (grant, biosketch, facilities, etc.)
            indexed: Whether document is indexed in RAG
            chunk_count: Number of chunks created

        Returns:
            doc_id: Unique document identifier
        """
        doc_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO project_documents
            (doc_id, project_id, filename, file_path, doc_type, indexed, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, project_id, filename, file_path, doc_type, int(indexed), chunk_count))

        self.conn.commit()
        logger.debug(f"Added document {filename} to project {project_id}")

        return doc_id

    def get_project_documents(self, project_id: str) -> List[Dict]:
        """
        Get all documents for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of document dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM project_documents
            WHERE project_id = ?
            ORDER BY uploaded_at DESC
        """, (project_id,))

        return [dict(row) for row in cursor.fetchall()]

    # ========== DRAFT OPERATIONS ==========

    def save_draft(
        self,
        project_id: str,
        section_name: str,
        content: str,
        citations: List[Dict],
        quality_score: float,
        quality_feedback: str = "",
        generation_params: Optional[Dict] = None,
        parent_draft_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Save a section draft (creates new version).

        Args:
            project_id: Project identifier
            section_name: Section name (e.g., "Background")
            content: Generated section text
            citations: List of citation dicts
            quality_score: Quality score (0.0-1.0)
            quality_feedback: Quality feedback text
            generation_params: Generation parameters (top_k, temperature, etc.)
            parent_draft_id: Parent draft ID for refinements
            conversation_history: Conversation history for multi-turn

        Returns:
            draft_id: Unique draft identifier
        """
        draft_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        # Get next version number
        cursor.execute("""
            SELECT COALESCE(MAX(version), 0) + 1 as next_version
            FROM section_drafts
            WHERE project_id = ? AND section_name = ?
        """, (project_id, section_name))
        version = cursor.fetchone()['next_version']

        # Insert draft
        cursor.execute("""
            INSERT INTO section_drafts
            (draft_id, project_id, section_name, version, content, citations,
             quality_score, quality_feedback, generation_params, parent_draft_id, conversation_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            draft_id,
            project_id,
            section_name,
            version,
            content,
            json.dumps(citations),
            quality_score,
            quality_feedback,
            json.dumps(generation_params or {}),
            parent_draft_id,
            json.dumps(conversation_history or [])
        ))

        self.conn.commit()
        logger.debug(f"Saved draft v{version} for {section_name} in project {project_id}")

        return draft_id

    def get_latest_draft(self, project_id: str, section_name: str) -> Optional[Dict]:
        """
        Get the latest draft for a section.

        Args:
            project_id: Project identifier
            section_name: Section name

        Returns:
            Draft dict or None if no drafts exist
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM section_drafts
            WHERE project_id = ? AND section_name = ?
            ORDER BY version DESC
            LIMIT 1
        """, (project_id, section_name))

        row = cursor.fetchone()

        if row:
            draft = dict(row)
            # Parse JSON fields
            draft['citations'] = json.loads(draft['citations']) if draft['citations'] else []
            draft['generation_params'] = json.loads(draft['generation_params']) if draft['generation_params'] else {}
            draft['conversation_history'] = json.loads(draft['conversation_history']) if draft['conversation_history'] else []
            return draft

        return None

    def get_draft_history(self, project_id: str, section_name: str) -> List[Dict]:
        """
        Get all drafts for a section (version history).

        Args:
            project_id: Project identifier
            section_name: Section name

        Returns:
            List of draft dicts, ordered by version
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM section_drafts
            WHERE project_id = ? AND section_name = ?
            ORDER BY version ASC
        """, (project_id, section_name))

        drafts = []
        for row in cursor.fetchall():
            draft = dict(row)
            draft['citations'] = json.loads(draft['citations']) if draft['citations'] else []
            draft['generation_params'] = json.loads(draft['generation_params']) if draft['generation_params'] else {}
            draft['conversation_history'] = json.loads(draft['conversation_history']) if draft['conversation_history'] else []
            drafts.append(draft)

        return drafts

    def get_draft_by_id(self, draft_id: str) -> Optional[Dict]:
        """
        Get a specific draft by ID.

        Args:
            draft_id: Draft identifier

        Returns:
            Draft dict or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM section_drafts WHERE draft_id = ?", (draft_id,))
        row = cursor.fetchone()

        if row:
            draft = dict(row)
            draft['citations'] = json.loads(draft['citations']) if draft['citations'] else []
            draft['generation_params'] = json.loads(draft['generation_params']) if draft['generation_params'] else {}
            draft['conversation_history'] = json.loads(draft['conversation_history']) if draft['conversation_history'] else []
            return draft

        return None

    # ========== FEEDBACK OPERATIONS ==========

    def add_feedback(self, draft_id: str, feedback_text: str) -> str:
        """
        Add user feedback for a draft.

        Args:
            draft_id: Draft identifier
            feedback_text: User's feedback text

        Returns:
            feedback_id: Unique feedback identifier
        """
        feedback_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO refinement_feedback (feedback_id, draft_id, feedback_text)
            VALUES (?, ?, ?)
        """, (feedback_id, draft_id, feedback_text))

        self.conn.commit()
        return feedback_id

    def get_draft_feedback(self, draft_id: str) -> List[Dict]:
        """
        Get all feedback for a draft.

        Args:
            draft_id: Draft identifier

        Returns:
            List of feedback dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM refinement_feedback
            WHERE draft_id = ?
            ORDER BY created_at ASC
        """, (draft_id,))

        return [dict(row) for row in cursor.fetchall()]

    # ========== TELEMETRY OPERATIONS ==========

    def log_generation(
        self,
        project_id: str,
        section_name: str,
        query: str,
        num_chunks: int,
        cache_hits: int,
        cache_misses: int,
        time_sec: float
    ):
        """
        Log a generation event for telemetry.

        Args:
            project_id: Project identifier
            section_name: Section name
            query: Generation query
            num_chunks: Number of chunks retrieved
            cache_hits: Embedding cache hits
            cache_misses: Embedding cache misses
            time_sec: Generation time in seconds
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO generation_log
            (log_id, project_id, section_name, query, num_chunks_retrieved,
             cache_hits, cache_misses, generation_time_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            project_id,
            section_name,
            query,
            num_chunks,
            cache_hits,
            cache_misses,
            time_sec
        ))

        self.conn.commit()

    def get_generation_logs(self, project_id: str) -> List[Dict]:
        """
        Get all generation logs for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of log dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM generation_log
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))

        return [dict(row) for row in cursor.fetchall()]

    # ========== UTILITY OPERATIONS ==========

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __del__(self):
        """Ensure connection is closed on deletion."""
        try:
            self.close()
        except:
            pass
