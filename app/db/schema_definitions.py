"""
Comprehensive database schema definitions for the organization_management database in the Bibliotheca system.
This file provides detailed schema information to help the LLM generate accurate SQL queries.
"""

SCHEMA_DEFINITIONS = {
    "report_management": {
        "description": "Database for storing event logs, usage statistics, and associated library hierarchy information.",
        "tables": {
            "events": {
                "name": "5",
                "description": "Stores aggregated event counts for a library system or location at specific timestamps.",
                "columns": [
                    {"name": "id", "type": "bigint", "primary_key": False, "description": "Unique identifier for the log entry"},
                    {"name": "eventTimestamp", "type": "timestamp with time zone", "description": "Timestamp for when the event counts were recorded or aggregated"},
                    {"name": "organizationId", "type": "uuid", "description": "Identifier for the library's parent organization (FK to hierarchyCaches.id)"},
                    {"name": "hierarchyId", "type": "uuid", "description": "Identifier for the specific library location itself (FK to hierarchyCaches.id)"},
                    {"name": "eventSrc", "type": "uuid", "description": "Identifier for the source system or device generating the event"},
                    {"name": "createdAt", "type": "timestamp with time zone", "description": "Timestamp when the log record was created"},
                    {"name": "updatedAt", "type": "timestamp with time zone", "description": "Timestamp when the log record was last updated"},
                    {"name": "1", "type": "integer", "description": "Total item(s) borrowed successfully in this period"},
                    {"name": "2", "type": "integer", "description": "Total item(s) borrowed unsuccessfully in this period"},
                    {"name": "3", "type": "integer", "description": "Total item(s) returned successfully in this period"},
                    {"name": "4", "type": "integer", "description": "Total item(s) returned unsuccessfully in this period"},
                    {"name": "5", "type": "integer", "description": "Total user(s) logged in successfully in this period"},
                    {"name": "6", "type": "integer", "description": "Total user(s) logged in unsuccessfully in this period"},
                    {"name": "7", "type": "integer", "description": "Total item(s) renewed successfully in this period"},
                    {"name": "8", "type": "integer", "description": "Total item(s) renewed unsuccessfully in this period"},
                    {"name": "32", "type": "integer", "description": "Total payment(s) made successfully in this period"},
                    {"name": "33", "type": "integer", "description": "Total payment(s) made unsuccessfully in this period"},
                    {"name": "38", "type": "integer", "description": "Total recommendation actions taken in this period"}
                ],
                "example_queries": []
            },
            "hierarchyCaches": {
                "name": "hierarchyCaches",
                "description": "Stores cached organizational hierarchy data (Library Systems, Libraries, Sub-Locations). Provides IDs, names, parentage, and path information. Used for joining event data (table '5') with hierarchy context.",
                "columns": [
                    {"name": "id", "type": "uuid", "primary_key": True, "description": "Unique identifier for the hierarchy (library system, library, or sub-location)"},
                    {"name": "name", "type": "VARCHAR(255)", "description": "Name of the library system, library, or sub-location"},
                    {"name": "shortName", "type": "VARCHAR(255)", "description": "Short name or abbreviation (Nullable)", "nullable": True},
                    {"name": "parentId", "type": "uuid", "foreign_key": "hierarchyCaches.id", "description": "Links to the parent hierarchy. NULL for top-level organizations.", "nullable": True},
                    {"name": "path", "type": "ARRAY", "description": "Path through hierarchy tree (Nullable)", "nullable": True},
                    {"name": "createdAt", "type": "TIMESTAMP WITH TIME ZONE", "description": "Timestamp when record was created"},
                    {"name": "updatedAt", "type": "TIMESTAMP WITH TIME ZONE", "description": "Timestamp when record was last updated"},
                    {"name": "deletedAt", "type": "TIMESTAMP WITH TIME ZONE", "description": "Timestamp when record was deleted (soft delete) (Nullable)", "nullable": True},
                    {"name": "lchierarchyId", "type": "uuid", "description": "Legacy hierarchy ID for migration purposes (Nullable)", "nullable": True}
                ],
                "example_queries": [
                    "SELECT hc.\"id\", hc.\"name\" FROM \"hierarchyCaches\" hc WHERE hc.\"deletedAt\" IS NULL",
                    "SELECT hc_loc.\"name\" FROM \"hierarchyCaches\" hc_loc JOIN \"hierarchyCaches\" hc_org ON hc_loc.\"parentId\" = hc_org.\"id\" WHERE hc_org.\"name\" = 'Example Library System' AND hc_loc.\"deletedAt\" IS NULL",
                    "SELECT hc.\"name\" FROM \"hierarchyCaches\" hc WHERE hc.\"parentId\" IS NULL AND hc.\"deletedAt\" IS NULL"
                ]
            },
            "footfall": {
                "name": "8",
                "description": "Stores footfall data (people entering/leaving) associated with specific device parts (e.g., gates) within a library location. Queries about general 'footfall' or 'visitors' should typically involve summing both column \"39\" (entries) and column \"40\" (exits).",
                "columns": [
                    {"name": "id", "type": "bigint", "primary_key": True, "description": "Unique identifier for the footfall log entry (Primary Key)"},
                    {"name": "eventTimestamp", "type": "timestamp with time zone", "description": "Timestamp when the footfall count was recorded"},
                    {"name": "organizationId", "type": "uuid", "foreign_key": "hierarchyCaches.id", "description": "Identifier for the library's parent organization", "nullable": True},
                    {"name": "hierarchyId", "type": "uuid", "foreign_key": "hierarchyCaches.id", "description": "Identifier for the specific library location", "nullable": True},
                    {"name": "eventSrc", "type": "uuid", "description": "Identifier for the source device generating the event", "nullable": True},
                    {"name": "partName", "type": "character varying", "description": "Name of the device part (e.g., gate) through which movement occurred", "nullable": False},
                    {"name": "instanceId", "type": "bigint", "description": "Specific instance identifier related to the event source or part", "nullable": False},
                    {"name": "createdAt", "type": "timestamp with time zone", "description": "Timestamp when the log record was created", "nullable": True},
                    {"name": "updatedAt", "type": "timestamp with time zone", "description": "Timestamp when the log record was last updated", "nullable": True},
                    {"name": "39", "type": "bigint", "description": "Cumulative count of people entering through this part at this timestamp (Default: 0)", "nullable": True, "default": "0"},
                    {"name": "40", "type": "bigint", "description": "Cumulative count of people exiting through this part at this timestamp (Default: 0)", "nullable": True, "default": "0"},
                    {"name": "41", "type": "bigint", "description": "Unknown/Unused footfall count? (Default: 0)", "nullable": True, "default": "0"},
                    {"name": "info", "type": "jsonb", "description": "Additional JSON details related to the footfall event", "nullable": True}
                ],
                "example_queries": [
                    "SELECT SUM(\"39\") AS total_entries FROM \"8\" WHERE \"hierarchyId\" = 'some-location-uuid' AND \"eventTimestamp\" >= NOW() - INTERVAL '1 day'",
                    "SELECT \"partName\", SUM(\"39\") AS entries, SUM(\"40\") AS exits FROM \"8\" WHERE \"hierarchyId\" = 'some-location-uuid' GROUP BY \"partName\""
                ]
            }
        }
    }
} 