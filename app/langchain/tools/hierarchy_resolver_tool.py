import logging
from typing import Type, List, Dict, Any, Optional
import sqlalchemy 
from sqlalchemy.engine import Connection 
from pydantic import BaseModel, Field 

from langchain_core.tools import BaseTool
from fuzzywuzzy import process # Using fuzzywuzzy for matching


from app.db.connection import get_db_engine 

logger = logging.getLogger(__name__)

# --- Input Schema ---
class HierarchyResolverInput(BaseModel):
    name_candidates: List[str] = Field(description="A list of potential hierarchy names (e.g., branch names, library names) mentioned by the user.")
    # organization_id: str = Field(description="The specific organization ID to scope the hierarchy search.") # Removed - Tool uses internal context
    # Optional: Add a threshold if you want the agent to control it, otherwise use a default
    # min_score_threshold: int = Field(default=85, description="Minimum confidence score (0-100) to consider a match valid.")

# --- Tool Implementation ---
class HierarchyNameResolverTool(BaseTool):
    """Tool to resolve potentially fuzzy user-provided hierarchy names (like branches, libraries)
    against the exact names stored in the 'hierarcyCaches' table for the organization associated with the request context.
    It uses fuzzy matching to find the best match and returns the exact database name, ID, and matching score."""
    name: str = "hierarchy_name_resolver"
    description: str = (
        "Resolves user-provided hierarchy entity names (e.g., 'Main Library', 'Argyle') against the exact names "
        "in the database for the relevant organization. Use this *before* querying data if the user mentions "
        "specific branches, libraries, or other hierarchy entities by name. Returns a mapping of input names "
        "to their resolved database name, ID, and matching score."
    )
    args_schema: Type[BaseModel] = HierarchyResolverInput
    # user_id: str # Removed
    organization_id: str # Passed during instantiation for context, **used internally**
    min_score_threshold: int = 85
    db_name: str = "report_management" # Assume we always use this DB for hierarchy

    def _run(self, name_candidates: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Synchronous execution. Mirrors the pattern in SQLQueryTool."""
        logger.warning("Running HierarchyNameResolverTool synchronously.")
        org_id_to_use = self.organization_id
        engine = get_db_engine(self.db_name)
        if not engine:
            error_msg = f"Database engine '{self.db_name}' not configured."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            with engine.connect() as connection:
                 return self._execute_logic(connection, name_candidates, org_id_to_use)
        except Exception as e:
            error_msg = f"Failed during sync execution: {str(e)}"
            logger.error(f"Error during synchronous hierarchy resolution: {e}", exc_info=True)
            raise ValueError(error_msg)

    async def _arun(self, name_candidates: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Resolve names asynchronously. Mirrors the pattern in SQLQueryTool's _run method structure."""
        org_id_to_use = self.organization_id
        logger.info(f"Executing Hierarchy Name Resolver for org {org_id_to_use} with candidates: {name_candidates}")
        engine = get_db_engine(self.db_name)
        if not engine:
            error_msg = f"Database engine '{self.db_name}' not configured."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            with engine.connect() as connection:
                return self._execute_logic(connection, name_candidates, org_id_to_use)
        except Exception as e:
            error_msg = f"Failed during async execution: {str(e)}"
            logger.error(f"Error during async hierarchy resolution: {e}", exc_info=True)
            raise ValueError(error_msg)

    # Renamed helper function for clarity, now takes the Connection object
    # Signature remains the same, it receives the org_id from _run/_arun
    def _execute_logic(self, connection: Connection, name_candidates: List[str], organization_id: str) -> Dict[str, Any]:
        """Core logic to fetch data and perform fuzzy matching using a SQLAlchemy Connection."""
        resolved_map: Dict[str, Dict[str, Any]] = {}
        hierarchy_cache_data: List[Dict[str, Any]] = []

        # 1. Fetch hierarchy names belonging ONLY to the specified organization OR its direct children
        try:
            # Select ID and Name for non-deleted entries where the ID is the org ID itself,
            # OR the parentId is the org ID (i.e., direct children/locations).
            query = sqlalchemy.text(
                'SELECT "id", "name" FROM "hierarchyCaches" ' \
                'WHERE "deletedAt" IS NULL AND (id = :org_id OR "parentId" = :org_id)'
            )
            # Pass the organization_id as a parameter
            result = connection.execute(query, {"org_id": organization_id})
            hierarchy_cache_data = [{"id": str(row.id), "name": row.name} for row in result.mappings()]
            logger.debug(f"Retrieved {len(hierarchy_cache_data)} hierarchy entries (org + direct children) for organization {organization_id}.")

            if not hierarchy_cache_data:
                 logger.warning(f"No hierarchy entries found for organization {organization_id} or its direct children. Cannot resolve names.")
                 for name in name_candidates:
                      resolved_map[name] = {"status": "no_hierarchy_data", "resolved_name": None, "id": None, "score": 0}
                 return {"resolution_results": resolved_map}

        except Exception as e:
            logger.error(f"Database error fetching hierarchy cache for organization {organization_id} and children: {e}", exc_info=True)
            db_error_msg = f"DB error fetching org/children cache: {str(e)}"
            for name in name_candidates:
                 resolved_map[name] = {"status": "error", "error_message": db_error_msg, "resolved_name": None, "id": None, "score": 0}
            return {"resolution_results": resolved_map, "error": f"Database error fetching org/children hierarchy data: {str(e)}"}

        # 2. Create lookups for matching
        # Exact (case-insensitive) lookup
        exact_match_lookup = {entry["name"].lower(): {"id": entry["id"], "name": entry["name"]} for entry in hierarchy_cache_data}
        # List for fuzzy matching
        fuzzy_match_name_list = [entry["name"] for entry in hierarchy_cache_data]
        fuzzy_match_id_lookup = {entry["name"]: entry["id"] for entry in hierarchy_cache_data}

        # 3. Process each candidate name
        for candidate in name_candidates:
            matched = False
            try:
                # Prioritize exact (case-insensitive) match
                exact_match_info = exact_match_lookup.get(candidate.lower())
                if exact_match_info:
                    matched_name = exact_match_info["name"]
                    matched_id = exact_match_info["id"]
                    logger.info(f"Resolved '{candidate}' to '{matched_name}' (ID: {matched_id}) via exact match.")
                    resolved_map[candidate] = {
                        "status": "found",
                        "resolved_name": matched_name,
                        "id": matched_id,
                        "score": 100 # Assign 100 for exact match
                    }
                    matched = True
                
                # If no exact match, try fuzzy matching
                if not matched:
                    best_match = process.extractOne(candidate, fuzzy_match_name_list, score_cutoff=self.min_score_threshold)
                    if best_match:
                        matched_name, score = best_match
                        matched_id = fuzzy_match_id_lookup[matched_name]
                        logger.info(f"Resolved '{candidate}' to '{matched_name}' (ID: {matched_id}) with fuzzy score {score}")
                        resolved_map[candidate] = {
                            "status": "found",
                            "resolved_name": matched_name,
                            "id": matched_id,
                            "score": score
                        }
                        matched = True
                    
                # If still no match (neither exact nor fuzzy above threshold)
                if not matched:
                    logger.warning(f"Could not resolve '{candidate}' via exact or fuzzy match (score >= {self.min_score_threshold})")
                    resolved_map[candidate] = {
                        "status": "not_found",
                        "resolved_name": None,
                        "id": None,
                        "score": 0
                    }

            except Exception as e:
                 logger.error(f"Error during matching process for candidate '{candidate}': {e}", exc_info=True)
                 resolved_map[candidate] = {
                     "status": "error",
                     "error_message": f"Matching error: {str(e)}",
                     "resolved_name": None,
                     "id": None,
                     "score": 0
                 }

        logger.debug(f"Hierarchy name resolution completed. Result map: {resolved_map}")
        # Final successful return structure
        return {"resolution_results": resolved_map}