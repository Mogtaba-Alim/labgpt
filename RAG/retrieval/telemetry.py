#!/usr/bin/env python3
"""
telemetry.py

Comprehensive retrieval telemetry and monitoring system.
Provides detailed logging, analytics, and performance tracking for RAG retrieval.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import uuid

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class QueryEvent:
    """Single query event for telemetry"""
    query_id: str
    timestamp: float
    original_query: str
    processed_queries: List[str]
    query_intent: str
    expansion_method: str
    
    # Retrieval metrics
    dense_candidates: List[str]  # chunk IDs
    sparse_candidates: List[str]  # chunk IDs
    fusion_method: str
    pre_rerank_scores: Dict[str, float]
    post_rerank_scores: Dict[str, float]
    final_chunk_ids: List[str]
    
    # Performance metrics
    query_processing_time: float
    dense_retrieval_time: float
    sparse_retrieval_time: float
    fusion_time: float
    reranking_time: float
    total_time: float
    
    # Quality metrics
    result_count: int
    score_distribution: Dict[str, float]  # min, max, mean, std
    diversity_score: float
    
    # User interaction (if available)
    user_feedback: Optional[str] = None
    click_positions: List[int] = field(default_factory=list)
    satisfaction_score: Optional[float] = None
    
    # System context
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    system_load: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    total_queries: int = 0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    hit_rate: float = 0.0
    avg_result_count: float = 0.0
    avg_diversity_score: float = 0.0
    
    dense_retrieval_time: float = 0.0
    sparse_retrieval_time: float = 0.0
    reranking_time: float = 0.0
    
    error_rate: float = 0.0
    timeout_rate: float = 0.0

class RetrievalTelemetry:
    """
    Comprehensive telemetry system for retrieval operations.
    
    Features:
    - Real-time query logging and metrics
    - Performance analytics and profiling
    - Retrieval method effectiveness tracking
    - User feedback integration
    - Configurable monitoring and alerting
    - Historical data analysis
    """
    
    def __init__(self,
                 telemetry_dir: str = "retrieval_telemetry",
                 buffer_size: int = 1000,
                 flush_interval: float = 60.0,
                 enable_real_time: bool = True):
        
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(exist_ok=True)
        
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.enable_real_time = enable_real_time
        
        # Storage files
        self.events_file = self.telemetry_dir / "query_events.jsonl"
        self.metrics_file = self.telemetry_dir / "metrics.json"
        self.config_file = self.telemetry_dir / "telemetry_config.json"
        
        # In-memory buffers
        self._event_buffer: deque = deque(maxlen=buffer_size)
        self._metrics_buffer: deque = deque(maxlen=buffer_size)
        
        # Real-time tracking
        self._active_queries: Dict[str, Dict[str, Any]] = {}
        self._response_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._total_queries = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._method_performance = defaultdict(list)
        self._query_patterns = defaultdict(int)
        self._user_satisfaction = deque(maxlen=500)
        
        # Periodic flushing
        if enable_real_time:
            self._setup_periodic_flush()
        
        # Load existing configuration
        self._load_config()
    
    def start_query(self, 
                   query_id: str,
                   original_query: str,
                   config: Optional[Dict[str, Any]] = None) -> None:
        """
        Start tracking a query
        
        Args:
            query_id: Unique identifier for the query
            original_query: Original query text
            config: Retrieval configuration used
        """
        with self._lock:
            self._active_queries[query_id] = {
                "start_time": time.time(),
                "original_query": original_query,
                "config": config or {},
                "events": []
            }
            
            self._total_queries += 1
            logger.debug(f"Started tracking query {query_id}")
    
    def log_query_processing(self,
                           query_id: str,
                           processed_queries: List[str],
                           query_intent: str,
                           expansion_method: str,
                           processing_time: float) -> None:
        """Log query processing step"""
        if query_id not in self._active_queries:
            logger.warning(f"Query {query_id} not found in active queries")
            return
        
        with self._lock:
            self._active_queries[query_id]["events"].append({
                "type": "query_processing",
                "processed_queries": processed_queries,
                "query_intent": query_intent,
                "expansion_method": expansion_method,
                "processing_time": processing_time
            })
    
    def log_dense_retrieval(self,
                          query_id: str,
                          candidate_ids: List[str],
                          scores: Dict[str, float],
                          retrieval_time: float) -> None:
        """Log dense retrieval step"""
        if query_id not in self._active_queries:
            return
        
        with self._lock:
            self._active_queries[query_id]["events"].append({
                "type": "dense_retrieval",
                "candidate_ids": candidate_ids,
                "scores": scores,
                "retrieval_time": retrieval_time
            })
    
    def log_sparse_retrieval(self,
                           query_id: str,
                           candidate_ids: List[str],
                           scores: Dict[str, float],
                           retrieval_time: float) -> None:
        """Log sparse retrieval step"""
        if query_id not in self._active_queries:
            return
        
        with self._lock:
            self._active_queries[query_id]["events"].append({
                "type": "sparse_retrieval",
                "candidate_ids": candidate_ids,
                "scores": scores,
                "retrieval_time": retrieval_time
            })
    
    def log_fusion(self,
                  query_id: str,
                  fusion_method: str,
                  pre_fusion_candidates: Dict[str, List[str]],
                  post_fusion_candidates: List[str],
                  fusion_time: float) -> None:
        """Log result fusion step"""
        if query_id not in self._active_queries:
            return
        
        with self._lock:
            self._active_queries[query_id]["events"].append({
                "type": "fusion",
                "fusion_method": fusion_method,
                "pre_fusion_candidates": pre_fusion_candidates,
                "post_fusion_candidates": post_fusion_candidates,
                "fusion_time": fusion_time
            })
    
    def log_reranking(self,
                     query_id: str,
                     pre_rerank_scores: Dict[str, float],
                     post_rerank_scores: Dict[str, float],
                     reranking_time: float) -> None:
        """Log re-ranking step"""
        if query_id not in self._active_queries:
            return
        
        with self._lock:
            self._active_queries[query_id]["events"].append({
                "type": "reranking",
                "pre_rerank_scores": pre_rerank_scores,
                "post_rerank_scores": post_rerank_scores,
                "reranking_time": reranking_time
            })
    
    def end_query(self,
                 query_id: str,
                 final_results: List[str],
                 total_time: Optional[float] = None) -> None:
        """
        End query tracking and generate event
        
        Args:
            query_id: Query identifier
            final_results: Final list of chunk IDs returned
            total_time: Total query time (calculated if None)
        """
        if query_id not in self._active_queries:
            logger.warning(f"Query {query_id} not found in active queries")
            return
        
        with self._lock:
            query_data = self._active_queries[query_id]
            
            if total_time is None:
                total_time = time.time() - query_data["start_time"]
            
            # Build complete query event
            event = self._build_query_event(query_id, query_data, final_results, total_time)
            
            # Add to buffer
            self._event_buffer.append(event)
            
            # Update real-time metrics
            self._update_real_time_metrics(event)
            
            # Clean up active query
            del self._active_queries[query_id]
            
            logger.debug(f"Completed tracking query {query_id} in {total_time:.3f}s")
    
    def log_error(self,
                 query_id: str,
                 error_type: str,
                 error_message: str,
                 error_time: Optional[float] = None) -> None:
        """Log query error"""
        with self._lock:
            self._error_count += 1
            
            error_event = {
                "query_id": query_id,
                "timestamp": time.time(),
                "error_type": error_type,
                "error_message": error_message,
                "error_time": error_time or time.time()
            }
            
            # Store error events separately
            self._event_buffer.append({"type": "error", "data": error_event})
            
            logger.warning(f"Query error {query_id}: {error_type} - {error_message}")
    
    def log_user_feedback(self,
                         query_id: str,
                         feedback_type: str,
                         feedback_data: Dict[str, Any]) -> None:
        """
        Log user feedback for a query
        
        Args:
            query_id: Query identifier
            feedback_type: Type of feedback (click, rating, etc.)
            feedback_data: Feedback details
        """
        feedback_event = {
            "query_id": query_id,
            "timestamp": time.time(),
            "feedback_type": feedback_type,
            "feedback_data": feedback_data
        }
        
        with self._lock:
            self._event_buffer.append({"type": "feedback", "data": feedback_event})
            
            # Update satisfaction tracking if rating provided
            if "satisfaction_score" in feedback_data:
                score = feedback_data["satisfaction_score"]
                if isinstance(score, (int, float)) and 0 <= score <= 5:
                    self._user_satisfaction.append(score)
    
    def get_real_time_metrics(self) -> PerformanceMetrics:
        """Get current real-time performance metrics"""
        with self._lock:
            if not self._response_times:
                return PerformanceMetrics()
            
            response_times = list(self._response_times)
            response_times.sort()
            
            metrics = PerformanceMetrics(
                total_queries=self._total_queries,
                avg_response_time=statistics.mean(response_times),
                p50_response_time=response_times[len(response_times) // 2],
                p95_response_time=response_times[int(len(response_times) * 0.95)],
                p99_response_time=response_times[int(len(response_times) * 0.99)],
                error_rate=self._error_count / max(self._total_queries, 1)
            )
            
            return metrics
    
    def get_method_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness metrics for different retrieval methods"""
        with self._lock:
            effectiveness = {}
            
            for method, times in self._method_performance.items():
                if times:
                    effectiveness[method] = {
                        "avg_time": statistics.mean(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "call_count": len(times)
                    }
            
            return effectiveness
    
    def get_query_patterns(self) -> Dict[str, int]:
        """Get common query patterns"""
        with self._lock:
            return dict(self._query_patterns)
    
    def get_user_satisfaction_metrics(self) -> Dict[str, float]:
        """Get user satisfaction metrics"""
        with self._lock:
            if not self._user_satisfaction:
                return {}
            
            satisfaction_scores = list(self._user_satisfaction)
            
            return {
                "avg_satisfaction": statistics.mean(satisfaction_scores),
                "median_satisfaction": statistics.median(satisfaction_scores),
                "satisfaction_count": len(satisfaction_scores),
                "high_satisfaction_rate": sum(1 for s in satisfaction_scores if s >= 4) / len(satisfaction_scores)
            }
    
    def analyze_query_performance(self, 
                                query_pattern: str,
                                time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Analyze performance for specific query patterns
        
        Args:
            query_pattern: Pattern to analyze (regex or substring)
            time_window: Time window for analysis (None for all time)
            
        Returns:
            Performance analysis results
        """
        # This would require loading and analyzing stored events
        # Implementation would depend on specific requirements
        return {
            "pattern": query_pattern,
            "analysis": "Implementation pending - requires event store analysis"
        }
    
    def export_telemetry_data(self, 
                            output_path: str,
                            time_range: Optional[Tuple[datetime, datetime]] = None,
                            query_filter: Optional[Callable] = None) -> bool:
        """
        Export telemetry data for analysis
        
        Args:
            output_path: Path for exported data
            time_range: Optional time range filter
            query_filter: Optional query filter function
            
        Returns:
            True if export successful
        """
        try:
            # Flush current buffer
            self._flush_events()
            
            # Load and filter events
            events = self._load_events(time_range, query_filter)
            
            # Export to file
            with open(output_path, 'w') as f:
                for event in events:
                    f.write(json.dumps(asdict(event) if hasattr(event, '__dict__') else event) + '\n')
            
            logger.info(f"Exported {len(events)} events to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _build_query_event(self,
                          query_id: str,
                          query_data: Dict[str, Any],
                          final_results: List[str],
                          total_time: float) -> QueryEvent:
        """Build complete query event from tracked data"""
        
        # Extract data from events
        processed_queries = []
        query_intent = "unknown"
        expansion_method = "none"
        dense_candidates = []
        sparse_candidates = []
        fusion_method = "none"
        pre_rerank_scores = {}
        post_rerank_scores = {}
        
        # Timing breakdown
        query_processing_time = 0.0
        dense_retrieval_time = 0.0
        sparse_retrieval_time = 0.0
        fusion_time = 0.0
        reranking_time = 0.0
        
        # Process events to extract metrics
        for event in query_data.get("events", []):
            event_type = event["type"]
            
            if event_type == "query_processing":
                processed_queries = event["processed_queries"]
                query_intent = event["query_intent"]
                expansion_method = event["expansion_method"]
                query_processing_time = event["processing_time"]
            
            elif event_type == "dense_retrieval":
                dense_candidates = event["candidate_ids"]
                dense_retrieval_time = event["retrieval_time"]
            
            elif event_type == "sparse_retrieval":
                sparse_candidates = event["candidate_ids"]
                sparse_retrieval_time = event["retrieval_time"]
            
            elif event_type == "fusion":
                fusion_method = event["fusion_method"]
                fusion_time = event["fusion_time"]
            
            elif event_type == "reranking":
                pre_rerank_scores = event["pre_rerank_scores"]
                post_rerank_scores = event["post_rerank_scores"]
                reranking_time = event["reranking_time"]
        
        # Calculate score distribution
        final_scores = [post_rerank_scores.get(chunk_id, 0.0) for chunk_id in final_results]
        score_distribution = {}
        if final_scores:
            score_distribution = {
                "min": min(final_scores),
                "max": max(final_scores),
                "mean": statistics.mean(final_scores),
                "std": statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
            }
        
        # Calculate diversity score (simple implementation)
        diversity_score = len(set(final_results)) / max(len(final_results), 1)
        
        return QueryEvent(
            query_id=query_id,
            timestamp=query_data["start_time"],
            original_query=query_data["original_query"],
            processed_queries=processed_queries,
            query_intent=query_intent,
            expansion_method=expansion_method,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            fusion_method=fusion_method,
            pre_rerank_scores=pre_rerank_scores,
            post_rerank_scores=post_rerank_scores,
            final_chunk_ids=final_results,
            query_processing_time=query_processing_time,
            dense_retrieval_time=dense_retrieval_time,
            sparse_retrieval_time=sparse_retrieval_time,
            fusion_time=fusion_time,
            reranking_time=reranking_time,
            total_time=total_time,
            result_count=len(final_results),
            score_distribution=score_distribution,
            diversity_score=diversity_score,
            retrieval_config=query_data.get("config", {})
        )
    
    def _update_real_time_metrics(self, event: QueryEvent) -> None:
        """Update real-time performance metrics"""
        self._response_times.append(event.total_time)
        
        # Update method performance
        self._method_performance["dense"].append(event.dense_retrieval_time)
        self._method_performance["sparse"].append(event.sparse_retrieval_time)
        self._method_performance["reranking"].append(event.reranking_time)
        
        # Update query patterns
        intent = event.query_intent
        self._query_patterns[intent] += 1
    
    def _flush_events(self) -> None:
        """Flush event buffer to disk"""
        if not self._event_buffer:
            return
        
        try:
            with open(self.events_file, 'a') as f:
                while self._event_buffer:
                    event = self._event_buffer.popleft()
                    f.write(json.dumps(asdict(event) if hasattr(event, '__dict__') else event) + '\n')
            
            logger.debug("Flushed event buffer to disk")
            
        except Exception as e:
            logger.error(f"Failed to flush events: {e}")
    
    def _setup_periodic_flush(self) -> None:
        """Setup periodic flushing of buffers"""
        def flush_worker():
            while True:
                time.sleep(self.flush_interval)
                self._flush_events()
        
        flush_thread = threading.Thread(target=flush_worker, daemon=True)
        flush_thread.start()
    
    def _load_config(self) -> None:
        """Load telemetry configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Apply configuration
                    self.buffer_size = config.get("buffer_size", self.buffer_size)
                    self.flush_interval = config.get("flush_interval", self.flush_interval)
        except Exception as e:
            logger.warning(f"Failed to load telemetry config: {e}")
    
    def _load_events(self,
                    time_range: Optional[Tuple[datetime, datetime]] = None,
                    query_filter: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Load events from storage with optional filtering"""
        events = []
        
        try:
            if self.events_file.exists():
                with open(self.events_file, 'r') as f:
                    for line in f:
                        event_data = json.loads(line.strip())
                        
                        # Apply time filter
                        if time_range:
                            event_time = datetime.fromtimestamp(event_data.get("timestamp", 0))
                            if not (time_range[0] <= event_time <= time_range[1]):
                                continue
                        
                        # Apply query filter
                        if query_filter and not query_filter(event_data):
                            continue
                        
                        events.append(event_data)
        
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
        
        return events
    
    def close(self) -> None:
        """Close telemetry system and flush remaining data"""
        logger.info("Closing telemetry system...")
        self._flush_events()
        
        # Save final metrics
        try:
            metrics = self.get_real_time_metrics()
            with open(self.metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save final metrics: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 