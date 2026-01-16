"""
Tools Module
Wrappers for DuckDB and ChromaDB operations
"""

import duckdb
import chromadb
from pathlib import Path
from typing import Optional, Dict, List, Any
from openai import OpenAI

from config import DUCKDB_PATH, CHROMA_PATH, OPENAI_API_KEY, EMBEDDING_MODEL


class DuckDBTool:
    """Tool for executing SQL queries on DuckDB"""

    def __init__(self, db_path: Path = DUCKDB_PATH):
        """Initialize DuckDB connection"""
        self.db_path = db_path
        self.con = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DuckDB database not found at {self.db_path}. "
                "Please run 'python -m src.data_setup' first."
            )
        self.con = duckdb.connect(str(self.db_path), read_only=False)

    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query and return results

        Args:
            sql: SQL query string

        Returns:
            Dict with 'success', 'data' (if successful), or 'error' (if failed)
        """
        try:
            result = self.con.execute(sql).fetchdf()
            return {
                "success": True,
                "data": result.to_dict('records'),
                "row_count": len(result)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql": sql
            }

    def get_schema(self) -> str:
        """
        Return database schema for use in prompts

        Returns:
            Formatted schema string with column names, types, and descriptions
        """
        schema_info = """
DATABASE SCHEMA:
Table: imdb

Columns:
- Poster_Link (TEXT): URL to movie poster image
- Series_Title (TEXT): Name of the movie
- Released_Year (INTEGER): Year the movie was released
- Certificate (TEXT): Movie rating certificate (e.g., 'A', 'UA', 'PG-13')
- Runtime (INTEGER): Movie runtime in minutes
- Genre (TEXT): Comma-separated genres (e.g., 'Action, Crime, Drama')
- IMDB_Rating (DOUBLE): IMDB rating (0-10 scale)
- Overview (TEXT): Movie plot summary
- Meta_score (DOUBLE): Metacritic score (0-100 scale)
- Director (TEXT): Name of the director
- Star1, Star2, Star3, Star4 (TEXT): Names of main cast members
- No_of_Votes (INTEGER): Total number of IMDB votes
- Gross (DOUBLE): Box office gross earnings in USD

Sample Data:
"""
        try:
            sample = self.con.execute(
                "SELECT Series_Title, Released_Year, Genre, IMDB_Rating, Director, Gross "
                "FROM imdb LIMIT 3"
            ).fetchdf()
            schema_info += sample.to_string(index=False)
        except Exception as e:
            schema_info += f"Error fetching sample: {e}"

        return schema_info

    def close(self):
        """Close database connection"""
        if self.con:
            self.con.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class ChromaDBTool:
    """Tool for semantic search using ChromaDB"""

    def __init__(self, persist_dir: Path = CHROMA_PATH, collection_name: str = "imdb_overviews"):
        """Initialize ChromaDB client and collection"""
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL
        self._connect()

    def _connect(self):
        """Establish ChromaDB connection"""
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {self.persist_dir}. "
                "Please run 'python -m src.data_setup' first."
            )

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found. "
                f"Please run 'python -m src.data_setup' first. Error: {e}"
            )

    def search(
        self,
        query: str,
        n_results: int = 10,
        where_filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search

        Args:
            query: Natural language search query
            n_results: Number of results to return
            where_filter: Optional metadata filter (e.g., {"Genre": {"$contains": "Comedy"}})

        Returns:
            Dict with 'success', 'movies' (list of results), or 'error'
        """
        try:
            query_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            query_embedding = query_response.data[0].embedding

            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }

            if where_filter:
                query_params["where"] = where_filter

            results = self.collection.query(**query_params)

            movies = []
            for i in range(len(results['ids'][0])):
                movie = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                movies.append(movie)

            return {
                "success": True,
                "movies": movies,
                "count": len(movies)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def search_with_structured_filter(
        self,
        query: str,
        movie_ids: List[str],
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search within a subset of movies (for hybrid queries)

        Args:
            query: Semantic search query
            movie_ids: List of movie IDs to search within
            n_results: Number of results to return

        Returns:
            Dict with search results
        """
        # ChromaDB doesn't support $in operator directly, so we'll search all
        # and filter in post-processing
        try:
            all_results = self.search(query, n_results=n_results * 3)

            if not all_results['success']:
                return all_results

            # Filter to only include movies in movie_ids
            filtered_movies = [
                movie for movie in all_results['movies']
                if movie['id'] in movie_ids
            ][:n_results]

            return {
                "success": True,
                "movies": filtered_movies,
                "count": len(filtered_movies)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()

            # Get sample documents
            sample = self.collection.peek(limit=3)

            return {
                "success": True,
                "stats": {
                    "total_documents": count,
                    "collection_name": self.collection_name,
                    "sample_titles": [
                        meta.get('Series_Title', 'Unknown')
                        for meta in sample['metadatas']
                    ]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instances for reuse
_duckdb_tool = None
_chromadb_tool = None


def get_duckdb_tool() -> DuckDBTool:
    """Get or create DuckDB tool instance"""
    global _duckdb_tool
    if _duckdb_tool is None:
        _duckdb_tool = DuckDBTool()
    return _duckdb_tool


def get_chromadb_tool() -> ChromaDBTool:
    """Get or create ChromaDB tool instance"""
    global _chromadb_tool
    if _chromadb_tool is None:
        _chromadb_tool = ChromaDBTool()
    return _chromadb_tool
