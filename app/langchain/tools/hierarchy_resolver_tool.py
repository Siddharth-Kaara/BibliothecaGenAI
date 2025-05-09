import logging
from typing import Type, List, Dict, Any, Optional, Tuple, ClassVar
import re
import sqlalchemy 
from sqlalchemy.ext.asyncio import AsyncConnection
from pydantic import BaseModel, Field 

from langchain_core.tools import BaseTool

# Add additional imports for better matching
from rapidfuzz import fuzz, process as rapidfuzz_process
from functools import partial

from app.db.connection import get_async_db_connection

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
    It uses advanced matching to find the best match and returns the exact database name, ID, and matching score."""
    name: str = "hierarchy_name_resolver"
    description: str = (
        "Resolves user-provided hierarchy entity names (e.g., 'Main Library', 'Argyle') against the exact names "
        "in the database for the relevant organization. Use this *before* querying data if the user mentions "
        "specific branches, libraries, or other hierarchy entities by name. Returns a mapping of input names "
        "to their resolved database name, ID, and matching score."
    )
    args_schema: Type[BaseModel] = HierarchyResolverInput
    organization_id: str # Passed during instantiation for context, **used internally**
    min_score_threshold: int = 85
    db_name: str = "report_management" # Assume we always use this DB for hierarchy
    
    # Common words to handle specially in library names - marked as ClassVar to avoid Pydantic treating it as a field
    COMMON_WORDS: ClassVar[List[str]] = ["library", "branch", "center", "location", "regional", "public", "community", "the"]
    
    # Priority branch names that should have special handling - common/important branch identifiers
    PRIORITY_NAMES: ClassVar[List[str]] = ["main", "central", "downtown", "headquarters", "hq", "primary"]
    
    # Advanced name processing
    def _preprocess_name(self, name: str) -> Dict[str, Any]:
        """Preprocess a name for matching. Returns dict with original, processed name, and extracted identifiers."""
        original = name
        processed = name.lower().strip()
        
        # Remove common suffixes/prefixes that might interfere with matching
        for word in self.COMMON_WORDS:
            processed = re.sub(rf'\b{word}\b', '', processed, flags=re.IGNORECASE).strip()
        
        # Extract core identifiers (don't strip common words from identifiers)
        identifiers = name.lower().strip()
        
        # Check for priority names
        has_priority = any(priority in processed for priority in self.PRIORITY_NAMES)
        
        return {
            "original": original,
            "processed": processed,
            "identifiers": identifiers,  # Now keeps full lowercase name as identifier
            "has_priority_name": has_priority
        }

    def _score_match(self, candidate_processed: Dict[str, Any], db_entry_processed: Dict[str, Any]) -> Dict[str, Any]:
        """Score how well a processed candidate matches a processed DB entry."""
        # Initialize result with no match
        result = {"match": False, "score": 0, "method": "no_match"}
        
        # Early exact match check (case-insensitive)
        if candidate_processed["processed"] == db_entry_processed["processed"]:
            return {"match": True, "score": 100, "method": "exact"}
        
        # Initialize scores dictionary for fuzzy matching
        scores = {}
        
        # Get the candidate and DB entry identifiers
        candidate_identifier = candidate_processed["identifiers"]
        db_identifier = db_entry_processed["identifiers"]
        
        # Calculate fuzzy match scores
        scores["ratio"] = fuzz.ratio(candidate_identifier, db_identifier)
        scores["partial"] = fuzz.partial_ratio(candidate_identifier, db_identifier)
        scores["token_set"] = fuzz.token_set_ratio(candidate_identifier, db_identifier)
        
        # Maximum score across all fuzzy metrics
        max_fuzzy_score = max(scores.values()) if scores else 0
        
        # For fuzzy matches, use a higher threshold if we're dealing with priority names
        fuzzy_threshold = self.min_score_threshold
        if candidate_processed["has_priority_name"] or db_entry_processed["has_priority_name"]:
            fuzzy_threshold = max(self.min_score_threshold + 5, 90)  # More strict for priority names
        
        # Only accept if above threshold
        if max_fuzzy_score >= fuzzy_threshold:
            best_method_key = max(scores.items(), key=lambda x: x[1])[0]
            current_method_for_log = f"fuzzy_{best_method_key}"

            # Stricter conditions for fuzzy matching
            is_short_term = len(candidate_identifier.split()) <= 1
            is_branch_term = "branch" in candidate_identifier.lower()
            
            # Special handling for branch names
            if is_branch_term:
                # For "X Branch" pattern, require higher score or exact match of the prefix
                branch_prefix = candidate_identifier.lower().replace(" branch", "").strip()
                db_prefix = db_identifier.lower().replace(" branch", "").strip()
                
                # If it's a specific branch name (e.g., "Main Branch", "DC Branch")
                if branch_prefix:
                    # Check if the prefix exactly matches
                    if branch_prefix == db_prefix:
                        result = {"match": True, "score": 100, "method": "branch_prefix_exact"}
                    # For partial matches on branch names, require very high score
                    elif best_method_key == "partial" and max_fuzzy_score < 95:
                        logger.debug(f"[{self.name}] Rejecting branch name '{candidate_identifier}' matching to '{db_identifier}' (score: {max_fuzzy_score:.2f}, method: {current_method_for_log})")
                        result = {"match": False, "score": max_fuzzy_score, "method": current_method_for_log + "_rejected_branch"}
                    else:
                        result = {
                            "match": True,
                            "score": max_fuzzy_score,
                            "method": current_method_for_log
                        }
            # For non-branch terms
            elif is_short_term and not candidate_processed["has_priority_name"] and max_fuzzy_score < 95:
                logger.debug(f"[{self.name}] Rejecting short term '{candidate_identifier}' matching to '{db_identifier}' (score: {max_fuzzy_score:.2f}, method: {current_method_for_log})")
                result = {"match": False, "score": max_fuzzy_score, "method": current_method_for_log + "_rejected_short"}
            else:
                result = {
                    "match": True,
                    "score": max_fuzzy_score,
                    "method": current_method_for_log
                }
        
        return result

    async def _run(self, name_candidates: List[str], **kwargs: Any) -> Dict[str, Any]:
        org_id_to_use = self.organization_id
        log_prefix = f"[Org: {org_id_to_use}] [HierarchyResolver] " # Added prefix
        logger.info(f"{log_prefix}Executing with {len(name_candidates)} candidates: {name_candidates[:5]}..." ) # Log start, limit candidates shown

        try:
            async with get_async_db_connection(self.db_name) as conn:
                results = await self._execute_logic(conn, name_candidates, org_id_to_use)
                found_count = sum(1 for r in results.get("resolution_results", {}).values() if r["status"] == "found")
                logger.info(f"{log_prefix}Completed. Found {found_count}/{len(name_candidates)} candidates.") # Log completion summary
                return results
        except Exception as e:
            error_msg = f"Failed during async execution: {str(e)}"
            logger.error(f"{log_prefix}Error: {error_msg}", exc_info=True) # Log error with prefix
            # Propagate error in a structured way if possible, or raise
            return {"resolution_results": {name: {"status": "error", "message": f"Tool error: {str(e)}"} for name in name_candidates}}
            # raise ValueError(error_msg) # Or re-raise if preferred

    async def _execute_logic(self, connection: AsyncConnection, name_candidates: List[str], organization_id: str) -> Dict[str, Any]:
        log_prefix = f"[Org: {organization_id}] [HierarchyResolver] " # Add prefix here too for consistency
        resolved_map: Dict[str, Dict[str, Any]] = {}
        hierarchy_cache_data: List[Dict[str, Any]] = []

        # 1. Fetch hierarchy names belonging ONLY to the specified organization OR its direct children
        try:
            # Select ID and Name for non-deleted entries where the ID is the org ID itself,
            # OR the parentId is the org ID (i.e., direct children/locations).
            query = sqlalchemy.text(
                'SELECT "id", "name", "shortName" FROM "hierarchyCaches" ' \
                'WHERE "deletedAt" IS NULL AND (id = :org_id OR "parentId" = :org_id)'
            )
            # Pass the organization_id as a parameter
            result = await connection.execute(query, {"org_id": organization_id})
            # Process the result mapping without awaiting
            mappings_result = result.mappings()
            hierarchy_cache_data = [
                {
                    "id": str(row.id), 
                    "name": row.name,
                    "shortName": row.shortName if hasattr(row, "shortName") else None
                } 
                for row in mappings_result
            ]
            
            logger.debug(f"{log_prefix}Retrieved {len(hierarchy_cache_data)} hierarchy entries.") # Keep DEBUG for data retrieval

            if not hierarchy_cache_data:
                 logger.warning(f"No hierarchy entries found for organization {organization_id} or its direct children. Cannot resolve names.")
                 for name in name_candidates:
                      resolved_map[name] = {"status": "no_hierarchy_data", "resolved_name": None, "id": None, "score": 0}
                 return {"resolution_results": resolved_map}

        except Exception as e:
            logger.error(f"{log_prefix}Database error fetching hierarchy cache: {e}", exc_info=True) # Log DB error
            db_error_msg = f"DB error fetching org/children cache: {str(e)}"
            for name in name_candidates:
                 resolved_map[name] = {"status": "error", "error_message": db_error_msg, "resolved_name": None, "id": None, "score": 0}
            return {"resolution_results": resolved_map, "error": f"Database error fetching org/children hierarchy data: {str(e)}"}

        # 2. Pre-process DB entries for better matching
        db_entries_processed = []
        for entry in hierarchy_cache_data:
            name_processed = self._preprocess_name(entry["name"])
            # Also process shortName if available
            shortname_processed = self._preprocess_name(entry["shortName"]) if entry["shortName"] else None
            
            db_entries_processed.append({
                "id": entry["id"],
                "name": entry["name"],
                "name_processed": name_processed,
                "shortname": entry["shortName"],
                "shortname_processed": shortname_processed
            })

        # 3. Process each candidate name
        for candidate in name_candidates:
            matched = False
            try:
                # Preprocess the candidate name
                candidate_processed = self._preprocess_name(candidate)
                
                # Score against all DB entries using our enhanced matching algorithm
                matches = []
                for db_entry in db_entries_processed:
                    # Try matching against full name first
                    match_result = self._score_match(candidate_processed, db_entry["name_processed"])
                    
                    # If shortname exists and no high-confidence match yet, also try matching against shortname
                    if db_entry["shortname_processed"] and (not match_result["match"] or match_result["score"] < 90):
                        shortname_match = self._score_match(candidate_processed, db_entry["shortname_processed"])
                        # Take the better of the two matches
                        if shortname_match["score"] > match_result["score"]:
                            match_result = shortname_match
                    
                    if match_result["match"]:
                        matches.append({
                            "db_entry": db_entry,
                            "score": match_result["score"],
                            "method": match_result["method"]
                        })
                
                # Find the best match if any
                if matches:
                    # Sort by score descending
                    matches.sort(key=lambda x: x["score"], reverse=True)
                    best_match = matches[0]
                    
                    matched_name = best_match["db_entry"]["name"]
                    matched_id = best_match["db_entry"]["id"]
                    match_score = best_match["score"]
                    match_method = best_match["method"]
                    
                    logger.debug(f"{log_prefix}Resolved '{candidate}' -> '{matched_name}' (Score: {match_score}, Method: {match_method})")
                    resolved_map[candidate] = {
                        "status": "found",
                        "resolved_name": matched_name,
                        "id": matched_id,
                        "score": match_score,
                        "method": match_method
                    }
                    matched = True
                    
                # If still no match
                if not matched:
                    logger.warning(f"{log_prefix}Could not resolve '{candidate}' (score >= {self.min_score_threshold})") # Keep WARN for failure
                    resolved_map[candidate] = {
                        "status": "not_found",
                        "resolved_name": None,
                        "id": None,
                        "score": 0
                    }

            except Exception as e:
                 logger.error(f"{log_prefix}Error during matching for '{candidate}': {e}", exc_info=True) # Log matching error
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
        
    # Implement ainvoke to ensure compatibility with BaseTool
    async def ainvoke(self, input_data: Dict[str, Any], **kwargs: Any) -> Any:
        """Override ainvoke to ensure it calls our async _run method."""
        validated_input = self.args_schema(**input_data)
        return await self._run(**validated_input.dict(), **kwargs)