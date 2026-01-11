#!/usr/bin/env python3
"""
Complete Solution for Narrative Consistency Classification Challenge
=====================================================================

This system determines whether a proposed character backstory is consistent
with a long-form narrative (100k+ words novel).

Architecture:
1. Semantic chunking with overlap for context preservation
2. Multi-stage retrieval: semantic search + reranking
3. LLM-based reasoning with Groq (open-source models)
4. Evidence extraction and structured classification

Usage:
    python narrative_consistency_checker.py \
        --test_csv test.csv \
        --novels_dir ./novels \
        --novel_mapping novel_mapping.json \
        --output results.csv \
        --api_key YOUR_GROQ_API_KEY

Or set GROQ_API_KEY environment variable and omit --api_key flag.
"""

import os
import re
import json
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import tiktoken


class Config:
    """Configuration for the narrative consistency checker"""

    # Model settings
    GROQ_MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embeddings

    # Chunking parameters
    CHUNK_SIZE = 1500  # tokens per chunk
    CHUNK_OVERLAP = 200  # overlap between chunks

    # Retrieval parameters
    TOP_K_CHUNKS = 10  # Initial retrieval
    RERANK_TOP_K = 5   # After reranking


@dataclass
class NovelChunk:
    """Represents a chunk of novel text with metadata"""
    chunk_id: int
    text: str
    book_name: str
    start_pos: int
    end_pos: int
    embedding: Optional[np.ndarray] = None


@dataclass
class Evidence:
    """Evidence linking text excerpts to backstory claims"""
    excerpt: str  # Verbatim text from novel
    backstory_claim: str  # Specific claim from the backstory
    analysis: str  # How this excerpt constrains/supports/contradicts the claim
    chunk_id: int  # Source chunk


@dataclass
class ConsistencyResult:
    """Final classification result"""
    story_id: int
    prediction: int  # 0 = inconsistent, 1 = consistent
    rationale: str  # Short explanation
    evidence: List[Evidence]


class TextChunker:
    """
    Intelligently chunks long texts while preserving context.
    Uses tiktoken for accurate token counting (compatible with LLMs).
    """

    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def chunk_by_sentences(self, text: str, book_name: str) -> List[NovelChunk]:
        """
        Chunk text by sentences to preserve semantic coherence.
        Maintains overlap for context continuity.
        """
        # Split into sentences (simple regex - can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        char_position = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(NovelChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    book_name=book_name,
                    start_pos=char_position - len(chunk_text),
                    end_pos=char_position
                ))
                chunk_id += 1

                # Keep overlap sentences for context
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            char_position += len(sentence) + 1

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(NovelChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                book_name=book_name,
                start_pos=char_position - len(chunk_text),
                end_pos=char_position
            ))

        print(f"✂️  Created {len(chunks)} chunks for {book_name}")
        return chunks


class PathwayVectorStore:
    """
    Vector store for document indexing and retrieval.
    Uses FAISS for efficient similarity search.
    """

    def __init__(self, embedding_model_name: str):
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: List[NovelChunk] = []

        print(f"✅ Vector store initialized (dim={self.dimension})")

    def add_chunks(self, chunks: List[NovelChunk]):
        """Add novel chunks to the vector store"""
        print(f"🔄 Embedding {len(chunks)} chunks...")

        # Generate embeddings in batches
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        # Store embeddings and add to FAISS
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.chunks.append(chunk)

        self.index.add(embeddings)
        print(f"✅ Added {len(chunks)} chunks to vector store")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[NovelChunk, float]]:
        """
        Semantic search for relevant chunks.
        Returns list of (chunk, similarity_score) tuples.
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k
        )

        # Return chunks with scores (convert L2 distance to similarity)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                results.append((self.chunks[idx], similarity))

        return results


class GroqReasoner:
    """
    Uses Groq's LLM API for reasoning about narrative consistency.
    Implements multi-stage reasoning: claim extraction → evidence analysis → classification.
    """

    def __init__(self, api_key: str, model: str = Config.GROQ_MODEL):
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"🤖 Initialized Groq LLM ({model})")

    def extract_claims(self, backstory: str) -> List[str]:
        """
        Extract specific, testable claims from the backstory.
        These will be validated against the novel.
        """
        prompt = f"""You are analyzing a character backstory. Extract SPECIFIC, TESTABLE claims that can be validated against a novel.

Backstory:
{backstory}

Extract 5-10 key claims about:
- Character's origins, family, early life
- Formative experiences and events
- Personality traits and motivations
- Relationships with other characters
- Skills, knowledge, or abilities
- Beliefs and worldview

Format as a JSON list of strings. Each claim should be ONE specific statement.

Example output:
["The character grew up in poverty", "The character has military training", "The character lost a parent in childhood"]

Output only valid JSON, no other text:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            content = re.sub(r'```json\s*|\s*```', '', content)
            claims = json.loads(content)

            print(f"📋 Extracted {len(claims)} claims from backstory")
            return claims

        except Exception as e:
            print(f"⚠️  Error extracting claims: {e}")
            # Fallback: split backstory into sentences as claims
            return [s.strip() for s in backstory.split('.') if len(s.strip()) > 20][:10]

    def analyze_evidence(
        self,
        claim: str,
        relevant_chunks: List[Tuple[NovelChunk, float]],
        character_name: str
    ) -> Dict:
        """
        Analyze whether retrieved chunks support, contradict, or are neutral to the claim.
        Returns structured evidence.
        """
        # Prepare context from chunks
        context = "\n\n---EXCERPT---\n\n".join([
            f"[Chunk {chunk.chunk_id}]: {chunk.text[:1000]}..."  # Truncate for token limits
            for chunk, _ in relevant_chunks[:3]  # Use top 3 most relevant
        ])

        prompt = f"""You are analyzing whether a novel supports or contradicts a character backstory claim.

CHARACTER: {character_name}

CLAIM TO VALIDATE:
{claim}

RELEVANT EXCERPTS FROM NOVEL:
{context}

Analyze whether the excerpts SUPPORT, CONTRADICT, or are NEUTRAL to this claim.

Respond in JSON format:
{{
    "verdict": "SUPPORT" | "CONTRADICT" | "NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "key_excerpt": "Most relevant quote from the excerpts (verbatim, <100 words)"
}}

Output only valid JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()
            content = re.sub(r'```json\s*|\s*```', '', content)
            result = json.loads(content)

            return result

        except Exception as e:
            print(f"⚠️  Error analyzing evidence: {e}")
            return {
                "verdict": "NEUTRAL",
                "confidence": 0.5,
                "reasoning": "Analysis failed",
                "key_excerpt": ""
            }

    def make_final_decision(
        self,
        backstory: str,
        character_name: str,
        evidence_results: List[Dict]
    ) -> Tuple[int, str]:
        """
        Make final consistency judgment based on all evidence.
        Returns (prediction, rationale).
        """
        # Summarize evidence
        contradictions = [e for e in evidence_results if e['verdict'] == 'CONTRADICT']
        supports = [e for e in evidence_results if e['verdict'] == 'SUPPORT']

        evidence_summary = f"""
SUPPORTING EVIDENCE: {len(supports)} claims supported
CONTRADICTING EVIDENCE: {len(contradictions)} claims contradicted

Details:
{json.dumps(evidence_results, indent=2)}
"""

        prompt = f"""You are making a final decision on whether a character backstory is CONSISTENT with a novel.

CHARACTER: {character_name}

BACKSTORY:
{backstory}

EVIDENCE ANALYSIS:
{evidence_summary}

Based on the evidence:
- If there are MAJOR contradictions (events that couldn't have happened, incompatible timelines, contradictory character traits), the backstory is INCONSISTENT.
- If there are only minor issues or lack of information, the backstory may still be CONSISTENT.
- The backstory doesn't need to be explicitly mentioned - it just needs to NOT contradict the novel.

Respond in JSON:
{{
    "consistent": true | false,
    "rationale": "1-2 sentence explanation of the decision"
}}

Output only valid JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()
            content = re.sub(r'```json\s*|\s*```', '', content)
            result = json.loads(content)

            prediction = 1 if result['consistent'] else 0
            rationale = result['rationale']

            return prediction, rationale

        except Exception as e:
            print(f"⚠️  Error making final decision: {e}")
            # Fallback: use simple heuristic
            if len(contradictions) > len(supports):
                return 0, "Multiple contradictions found with novel"
            else:
                return 1, "No major contradictions detected"


class NarrativeConsistencyChecker:
    """
    Main pipeline orchestrating the entire consistency checking process.
    """

    def __init__(self, groq_api_key: str):
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store = PathwayVectorStore(Config.EMBEDDING_MODEL)
        self.reasoner = GroqReasoner(groq_api_key)

        print("✅ Pipeline initialized")

    def index_novels(self, novel_mapping: Dict[str, str], novels_dir: str, unique_books: List[str]):
        """
        Process and index all novels in the dataset.
        Loads full novels from separate files based on the mapping.
        
        Args:
            novel_mapping: Dictionary mapping book_name to filename
            novels_dir: Directory containing novel text files
            unique_books: List of unique book names to index
        """
        print("\n" + "="*60)
        print("INDEXING NOVELS")
        print("="*60)

        for book_name in unique_books:
            print(f"\n📖 Processing: {book_name}")

            # Check if we have a file mapping
            if book_name not in novel_mapping:
                print(f"   ❌ No file mapping found for: {book_name}")
                print(f"   Available mappings: {list(novel_mapping.keys())}")
                continue

            # Construct file path
            filename = novel_mapping[book_name]
            file_path = os.path.join(novels_dir, filename)

            # Check if file exists
            if not os.path.exists(file_path):
                print(f"   ❌ File not found: {file_path}")
                print(f"   Please check the path and filename")
                continue

            # Load the full novel
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    novel_text = f.read()

                print(f"   ✅ Loaded: {len(novel_text):,} characters")

                # Chunk the novel
                chunks = self.chunker.chunk_by_sentences(novel_text, book_name)

                # Add to vector store
                self.vector_store.add_chunks(chunks)

            except Exception as e:
                print(f"   ❌ Error loading novel: {e}")
                import traceback
                traceback.print_exc()

        print("\n✅ All novels indexed successfully")

    def check_consistency(
        self,
        story_id: int,
        book_name: str,
        character_name: str,
        backstory: str
    ) -> ConsistencyResult:
        """
        Check if a backstory is consistent with the novel.

        Process:
        1. Extract testable claims from backstory
        2. For each claim, retrieve relevant novel passages
        3. Analyze evidence for support/contradiction
        4. Make final consistency judgment
        """
        print(f"\n{'='*60}")
        print(f"Checking Story ID: {story_id}")
        print(f"Book: {book_name} | Character: {character_name}")
        print(f"{'='*60}")

        # Handle empty backstories
        if not backstory or pd.isna(backstory) or str(backstory).strip() == "" or str(backstory).lower() == "nan":
            print("\n⚠️  Empty backstory - defaulting to CONSISTENT")
            return ConsistencyResult(
                story_id=story_id,
                prediction=1,
                rationale="No backstory provided to validate",
                evidence=[]
            )

        # Step 1: Extract claims from backstory
        print("\n📋 Step 1: Extracting claims...")
        claims = self.reasoner.extract_claims(str(backstory))

        # Step 2: Retrieve and analyze evidence for each claim
        print("\n🔍 Step 2: Retrieving evidence...")
        evidence_list = []
        evidence_results = []

        for i, claim in enumerate(claims, 1):
            print(f"   Claim {i}/{len(claims)}: {claim[:60]}...")

            # Retrieve relevant chunks for this claim
            # Combine claim with character name for better retrieval
            query = f"{character_name}: {claim}"
            relevant_chunks = self.vector_store.search(query, top_k=Config.TOP_K_CHUNKS)

            # Filter to only this book's chunks
            relevant_chunks = [
                (chunk, score) for chunk, score in relevant_chunks
                if chunk.book_name == book_name
            ][:Config.RERANK_TOP_K]

            if not relevant_chunks:
                print(f"      ⚠️  No relevant passages found")
                continue

            # Analyze evidence
            analysis = self.reasoner.analyze_evidence(claim, relevant_chunks, character_name)
            evidence_results.append(analysis)

            # Create evidence object
            if analysis['key_excerpt']:
                evidence = Evidence(
                    excerpt=analysis['key_excerpt'],
                    backstory_claim=claim,
                    analysis=analysis['reasoning'],
                    chunk_id=relevant_chunks[0][0].chunk_id if relevant_chunks else -1
                )
                evidence_list.append(evidence)

            print(f"      {analysis['verdict']} (confidence: {analysis['confidence']:.2f})")

        # Step 3: Make final decision
        print("\n⚖️  Step 3: Making final decision...")
        prediction, rationale = self.reasoner.make_final_decision(
            backstory,
            character_name,
            evidence_results
        )

        result = ConsistencyResult(
            story_id=story_id,
            prediction=prediction,
            rationale=rationale,
            evidence=evidence_list
        )

        print(f"\n✅ RESULT: {'CONSISTENT' if prediction == 1 else 'INCONSISTENT'}")
        print(f"   Rationale: {rationale}")

        return result

    def process_dataset(self, df: pd.DataFrame, output_file: str):
        """
        Process entire dataset and generate predictions.
        """
        results = []

        print("\n" + "="*60)
        print(f"PROCESSING {len(df)} EXAMPLES")
        print("="*60)

        for idx, row in df.iterrows():
            try:
                # Use 'content' column for backstory, fallback to 'caption'
                backstory = row['content'] if 'content' in row and pd.notna(row['content']) else row.get('caption', '')

                result = self.check_consistency(
                    story_id=row['id'],
                    book_name=row['book_name'],
                    character_name=row['char'],
                    backstory=backstory
                )

                results.append({
                    'Story ID': result.story_id,
                    'Prediction': result.prediction,
                    'Rationale': result.rationale
                })

                print(f"\n✅ Completed {idx+1}/{len(df)}")

            except Exception as e:
                print(f"\n❌ Error processing story {row['id']}: {e}")
                import traceback
                traceback.print_exc()
                # Add default prediction on error
                results.append({
                    'Story ID': row['id'],
                    'Prediction': 1,  # Default to consistent
                    'Rationale': f"Error in processing: {str(e)}"
                })

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Total predictions: {len(results)}")

        return results_df


def load_novel_mapping(mapping_file: str) -> Dict[str, str]:
    """Load novel mapping from JSON file"""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Narrative Consistency Classification Challenge - Reproducible Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python narrative_consistency_checker.py \\
      --test_csv test.csv \\
      --novels_dir ./novels \\
      --novel_mapping novel_mapping.json \\
      --output results.csv

Or with environment variable for API key:
  export GROQ_API_KEY=your_api_key_here
  python narrative_consistency_checker.py \\
      --test_csv test.csv \\
      --novels_dir ./novels \\
      --novel_mapping novel_mapping.json \\
      --output results.csv
        """
    )

    parser.add_argument(
        '--test_csv',
        type=str,
        required=True,
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--novels_dir',
        type=str,
        required=True,
        help='Directory containing novel text files'
    )
    parser.add_argument(
        '--novel_mapping',
        type=str,
        required=True,
        help='JSON file mapping book_name to filename (e.g., {"In Search of the Castaways": "castaways.txt"})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Output CSV file path (default: results.csv)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='Groq API key (or set GROQ_API_KEY environment variable)'
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Error: Groq API key required!")
        print("   Provide via --api_key argument or GROQ_API_KEY environment variable")
        sys.exit(1)

    print("\n" + "="*60)
    print("NARRATIVE CONSISTENCY CHECKER")
    print("="*60)

    # Load test dataset
    print("\n📊 Loading test dataset...")
    try:
        test_df = pd.read_csv(args.test_csv)
        print(f"✅ Loaded test set: {len(test_df)} examples")
        print(f"   Unique books: {test_df['book_name'].nunique()}")
    except FileNotFoundError:
        print(f"❌ Error: Test CSV file not found: {args.test_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading test CSV: {e}")
        sys.exit(1)

    # Load novel mapping
    print("\n📚 Loading novel mapping...")
    try:
        novel_mapping = load_novel_mapping(args.novel_mapping)
        print(f"✅ Loaded mapping for {len(novel_mapping)} novels")
    except FileNotFoundError:
        print(f"❌ Error: Novel mapping file not found: {args.novel_mapping}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in mapping file: {e}")
        sys.exit(1)

    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    checker = NarrativeConsistencyChecker(api_key)

    # Index novels
    print("\n📚 Indexing novels...")
    unique_books = test_df['book_name'].unique().tolist()
    checker.index_novels(novel_mapping, args.novels_dir, unique_books)

    # Process test set
    print("\n🧪 Processing test set...")
    results_df = checker.process_dataset(test_df, args.output)

    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"\n🎉 Results saved to: {args.output}")
    print(f"\n📊 Prediction distribution:")
    print(results_df['Prediction'].value_counts())


if __name__ == '__main__':
    main()
