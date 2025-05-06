#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

# Optional: Database Readiness Check (Simple TCP check)
# Extract host and port from the first DB URL (adjust parsing if format differs)
# Assumes format like: "db_name=postgresql://user:pass@host:port/db"
if [ -n "$DATABASE_URLS" ]; then
    DB_URL=$(echo $DATABASE_URLS | cut -d',' -f1 | cut -d'=' -f2-)
    DB_HOST=$(echo $DB_URL | sed -n 's_.*@\\([^:]*\\):.*_\\1_p')
    DB_PORT=$(echo $DB_URL | sed -n 's_.*:\\([0-9]*\\)/.*_\\1_p')

    if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
        echo "Waiting for database at $DB_HOST:$DB_PORT..."
        # Use netcat (nc) if available, otherwise fallback or skip
        # Add `apt-get install -y netcat-openbsd` to Dockerfile if using this check
        # while ! nc -z $DB_HOST $DB_PORT; do
        #   sleep 1
        # done
        # Alternative using bash built-in (might not be available in sh/alpine):
        # while ! timeout 1 bash -c "</dev/tcp/$DB_HOST/$DB_PORT"; do 
        #    sleep 1
        # done
        # Simple sleep as a fallback for now if nc isn't installed
        sleep 5 
        echo "Database connection attempt assumed possible."
    else
        echo "Could not parse DB_HOST/DB_PORT from DATABASE_URLS for readiness check."
        sleep 5 # Still wait a bit
    fi
else
    echo "DATABASE_URLS not set, skipping database readiness check."
    sleep 2
fi


# Run Alembic Migrations (Commented out as Alembic is not used)
# echo "Running database migrations..."
# alembic upgrade head

# Start Uvicorn Server
echo "Starting Uvicorn server..."
echo "Current PATH: $PATH"
echo "Locating uvicorn: $(which uvicorn || echo 'uvicorn not found by which')"
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 80 \
    # Optimal worker count formula
    --workers $(( $(nproc) * 2 + 1 )) \
    --timeout-keep-alive 30 \
    --limit-concurrency 1000 \
    --limit-max-requests 10000
