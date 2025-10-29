#!/bin/bash
# Comprehensive RAG Pipeline Feature Test Script
# Tests all features using CLI commands

# Exit on any error
set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  LabGPT RAG Pipeline - Feature Test"
echo "=========================================="
echo ""

# Function to handle errors
handle_error() {
    echo -e "${RED}✗ Test failed at: $1${NC}"
    echo ""
    echo "Please check the error messages above."
    echo "Make sure you're running from the labgpt repository root directory."
    exit 1
}

# Set error trap
trap 'handle_error "${BASH_COMMAND}"' ERR

# Check if corpus path provided
if [ -z "$1" ]; then
    echo "Usage: bash test_rag_features.sh <path_to_corpus> [test_query]"
    echo ""
    echo "Example:"
    echo "  bash test_rag_features.sh ../sample_papers/ \"machine learning\""
    echo "  bash test_rag_features.sh /path/to/documents/ \"your query here\""
    exit 1
fi

CORPUS_PATH="$1"
TEST_QUERY="${2:-machine learning}"
INDEX_DIR="test_rag_index"

echo "Corpus: $CORPUS_PATH"
echo "Test Query: '$TEST_QUERY'"
echo "Index Directory: $INDEX_DIR"
echo ""

# Clean up any existing test index
if [ -d "$INDEX_DIR" ]; then
    echo "Cleaning up existing test index..."
    rm -rf "$INDEX_DIR"
    echo ""
fi

# Test 1: Basic Ingestion (Default Preset)
echo -e "${BLUE}=========================================="
echo "TEST 1: Basic Ingestion (Default Preset)"
echo -e "==========================================${NC}"
python -m RAG.cli ingest --docs "$CORPUS_PATH" --index "$INDEX_DIR"
echo -e "${GREEN}✓ Test 1 Complete${NC}"
echo ""

# Test 2: Status Check
echo -e "${BLUE}=========================================="
echo "TEST 2: Status Check"
echo -e "==========================================${NC}"
python -m RAG.cli status --index "$INDEX_DIR"
echo -e "${GREEN}✓ Test 2 Complete${NC}"
echo ""

# Test 3: Basic Search
echo -e "${BLUE}=========================================="
echo "TEST 3: Basic Search (No Enhancements)"
echo -e "==========================================${NC}"
python -m RAG.cli ask --index "$INDEX_DIR" --query "$TEST_QUERY" --top-k 5
echo -e "${GREEN}✓ Test 3 Complete${NC}"
echo ""

# Test 4: Search with Query Expansion
echo -e "${BLUE}=========================================="
echo "TEST 4: Search with Query Expansion (PRF)"
echo -e "==========================================${NC}"
python -m RAG.cli ask --index "$INDEX_DIR" --query "$TEST_QUERY" --expand --top-k 10
echo -e "${GREEN}✓ Test 4 Complete${NC}"
echo ""

# Test 5: Search with Cited Spans
echo -e "${BLUE}=========================================="
echo "TEST 5: Search with Cited Spans"
echo -e "==========================================${NC}"
python -m RAG.cli ask --index "$INDEX_DIR" --query "$TEST_QUERY" --cited-spans --top-k 3
echo -e "${GREEN}✓ Test 5 Complete${NC}"
echo ""

# Test 6: All Features Combined
echo -e "${BLUE}=========================================="
echo "TEST 6: All Features Combined"
echo -e "==========================================${NC}"
python -m RAG.cli ask --index "$INDEX_DIR" --query "$TEST_QUERY" --expand --cited-spans --top-k 5
echo -e "${GREEN}✓ Test 6 Complete${NC}"
echo ""

# Test 7: Create Snapshot
echo -e "${BLUE}=========================================="
echo "TEST 7: Create Reproducibility Snapshot"
echo -e "==========================================${NC}"
python -m RAG.cli snapshot --index "$INDEX_DIR"
echo -e "${GREEN}✓ Test 7 Complete${NC}"
echo ""

# Test 8: Status Check (with snapshot info)
echo -e "${BLUE}=========================================="
echo "TEST 8: Status Check (After Snapshot)"
echo -e "==========================================${NC}"
python -m RAG.cli status --index "$INDEX_DIR"
echo -e "${GREEN}✓ Test 8 Complete${NC}"
echo ""

# Test 9: Re-ingestion (Cache Hit Test)
echo -e "${BLUE}=========================================="
echo "TEST 9: Re-ingestion (Embedding Cache Test)"
echo -e "==========================================${NC}"
echo -e "${YELLOW}Re-ingesting same documents to test cache performance...${NC}"
echo -e "${YELLOW}Look for high cache hit percentage in output!${NC}"
echo ""
python -m RAG.cli ingest --docs "$CORPUS_PATH" --index "$INDEX_DIR"
echo -e "${GREEN}✓ Test 9 Complete${NC}"
echo ""

# Test 10: Research Preset
echo -e "${BLUE}=========================================="
echo "TEST 10: Research Preset (Enhanced Features)"
echo -e "==========================================${NC}"
INDEX_RESEARCH="${INDEX_DIR}_research"
python -m RAG.cli ingest --docs "$CORPUS_PATH" --index "$INDEX_RESEARCH" --preset research
echo -e "${GREEN}✓ Test 10 Complete${NC}"
echo ""

# Test 11: Search with Research Preset
echo -e "${BLUE}=========================================="
echo "TEST 11: Search with Research Preset"
echo -e "==========================================${NC}"
python -m RAG.cli ask --index "$INDEX_RESEARCH" --query "$TEST_QUERY" --expand --top-k 10
echo -e "${GREEN}✓ Test 11 Complete${NC}"
echo ""

# Final Summary
echo -e "${GREEN}=========================================="
echo "  ALL TESTS COMPLETED SUCCESSFULLY!"
echo "==========================================${NC}"
echo ""
echo "Features Tested:"
echo "  ✓ Basic ingestion (default preset)"
echo "  ✓ Status checking"
echo "  ✓ Basic search (FAISS + BM25 + RRF fusion)"
echo "  ✓ Query expansion (PRF-style)"
echo "  ✓ Cited span extraction"
echo "  ✓ Combined features"
echo "  ✓ Reproducibility snapshots"
echo "  ✓ Embedding cache (SHA256-based)"
echo "  ✓ Research preset (all enhancements)"
echo ""
echo "Output directories created:"
echo "  - $INDEX_DIR/"
echo "  - ${INDEX_RESEARCH}/"
echo ""
echo "To clean up test outputs:"
echo "  rm -rf $INDEX_DIR ${INDEX_RESEARCH}"
echo ""