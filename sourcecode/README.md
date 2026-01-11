# Narrative Consistency Classification Challenge

A robust, reproducible solution for determining whether a proposed character backstory is consistent with a long-form narrative (100k+ words novel).

## 🎯 Overview

This system uses advanced NLP techniques to validate character backstories against full-length novels:

- **Semantic Chunking**: Intelligently chunks novels while preserving context
- **Vector Embeddings**: Uses sentence-transformers for semantic search
- **FAISS**: Efficient similarity search for relevant passages
- **LLM Reasoning**: Multi-stage reasoning with Groq API for claim extraction, evidence analysis, and classification

## ✨ Features

- ✅ **Fully Automated**: No manual steps, runs end-to-end from command-line
- ✅ **Reproducible**: All inputs via CLI arguments, no interactive prompts
- ✅ **Robust Validation**: Comprehensive input validation and error handling
- ✅ **Scalable**: Efficiently processes large novels (100k+ words)
- ✅ **Well-Documented**: Clear usage instructions and examples

## 📋 Requirements

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))
- Input CSV file with test data
- Novel text files (.txt format)
- Novel mapping JSON file

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd narrative-consistency-classifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

You need three files:

1. **Test CSV** (`test.csv`): Contains columns `id`, `book_name`, `char`, `content` (or `caption`)
2. **Novels Directory**: Directory containing novel text files
3. **Novel Mapping** (`novel_mapping.json`): Maps book names to filenames

Example `novel_mapping.json`:
```json
{
  "In Search of the Castaways": "In search of the castaways.txt",
  "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
}
```

### 3. Set Up API Key

Get your Groq API key from [Groq Console](https://console.groq.com/) and set it:

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**Linux/Mac:**
```bash
export GROQ_API_KEY=your_api_key_here
```

### 4. Run the Script

```bash
python narrative_consistency_checker.py \
    --test_csv test.csv \
    --novels_dir ./novels \
    --novel_mapping novel_mapping.json \
    --output results.csv
```

Or with API key as command-line argument:

```bash
python narrative_consistency_checker.py \
    --test_csv test.csv \
    --novels_dir ./novels \
    --novel_mapping novel_mapping.json \
    --output results.csv \
    --api_key YOUR_API_KEY
```

## 📖 Usage

### Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--test_csv` | Yes | Path to test CSV file | `--test_csv test.csv` |
| `--novels_dir` | Yes | Directory containing novel text files | `--novels_dir ./novels` |
| `--novel_mapping` | Yes | JSON file mapping book names to filenames | `--novel_mapping mapping.json` |
| `--output` | No | Output CSV file path (default: `results.csv`) | `--output results.csv` |
| `--api_key` | Yes* | Groq API key (or use `GROQ_API_KEY` env var) | `--api_key key123` |

*Required unless `GROQ_API_KEY` environment variable is set.

### Input Format

**Test CSV** should contain:
- `id`: Story ID (integer)
- `book_name`: Name of the book (must match keys in `novel_mapping.json`)
- `char`: Character name (string)
- `content` or `caption`: Character backstory text (string)

Example:
```csv
id,book_name,char,content
95,The Count of Monte Cristo,Noirtier,The character was a wealthy aristocrat...
```

**Novel Text Files**: Plain text files (`.txt`) containing the full novel text. Filenames must match values in `novel_mapping.json`.

### Output Format

The script generates a CSV file with predictions:

```csv
Story ID,Prediction,Rationale
95,1,No major contradictions found with the novel
96,0,Multiple contradictions detected in character background
```

- `Prediction`: `0` (inconsistent) or `1` (consistent)
- `Rationale`: Brief explanation of the decision

## 📁 Project Structure

```
.
├── narrative_consistency_checker.py  # Main script
├── requirements.txt                  # Python dependencies
├── novel_mapping.json.example        # Example mapping file
├── README.md                         # This file
├── README_REPRODUCIBLE.md           # Detailed reproducible guide
├── RUN_EXAMPLE.md                    # CLI usage examples
└── .vscode/                          # VS Code configuration
    ├── launch.json                   # Debug configurations
    └── settings.json                 # Python settings
```

## 🔧 How It Works

1. **Indexing Phase**:
   - Loads novel text files
   - Chunks novels into overlapping semantic segments
   - Generates embeddings for each chunk
   - Builds FAISS index for efficient search

2. **Processing Phase** (for each test example):
   - Extracts testable claims from the backstory
   - Retrieves relevant novel passages using semantic search
   - Analyzes evidence (support/contradict/neutral) for each claim
   - Makes final consistency judgment using LLM reasoning

3. **Output**:
   - Generates predictions with rationales
   - Saves results to CSV file

## 🛠️ Configuration

Default configuration (can be modified in code):

```python
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500  # tokens per chunk
CHUNK_OVERLAP = 200  # overlap between chunks
TOP_K_CHUNKS = 10  # Initial retrieval
RERANK_TOP_K = 5   # After reranking
```

## 📊 Performance Notes

- **First run**: Downloads embedding model (~80MB)
- **Indexing**: Takes time proportional to novel length
- **Processing**: ~10-30 seconds per example (depends on API response time)
- **Memory**: Loads all novels into memory during indexing

## 🐛 Troubleshooting

### File Not Found Errors

- Verify all file paths are correct (use absolute paths if needed)
- Check that novel filenames in mapping match actual files
- Ensure book names in CSV match keys in mapping file

### API Key Errors

- Verify your Groq API key is correct
- Check API rate limits if you encounter API errors
- Ensure internet connection for API calls

### Memory Issues

- The script loads all novels into memory during indexing
- For very large novels, ensure sufficient RAM available
- Processing is sequential, so runtime memory is manageable

## 📝 Example Output

```
============================================================
NARRATIVE CONSISTENCY CHECKER
============================================================

📋 Input Parameters:
   Test CSV: test.csv
   Novels Directory: ./novels
   Novel Mapping: novel_mapping.json
   Output File: results.csv
   API Key: ******************** (hidden)

🔍 Validating inputs...
✅ All input files/directories validated

📊 Loading test dataset...
✅ Loaded test set: 60 examples
   Unique books: 2

📚 Loading novel mapping...
✅ Loaded mapping for 2 novels
   Books: In Search of the Castaways, The Count of Monte Cristo

🔧 Initializing pipeline...
✅ Pipeline initialized

📚 Indexing novels...
✂️  Created 152 chunks for In Search of the Castaways
✅ Added 152 chunks to vector store
✂️  Created 502 chunks for The Count of Monte Cristo
✅ Added 502 chunks to vector store

🧪 Processing test set...
[... processing continues ...]

✅ Results saved to: results.csv
```

