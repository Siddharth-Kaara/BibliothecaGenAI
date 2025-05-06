# Stage 1: Build Stage - Installs dependencies and build tools if needed
FROM python:3.12.9-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Set environment variables for pip and python
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Optional: Install OS packages if needed for compiling dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Copy the application code
COPY ./app /app/app
# Removed static copy

# Stage 2: Final Stage - Copies application code and dependencies from builder
FROM python:3.12.9-slim-bookworm

# Set working directory
WORKDIR /app

# Add the non-root user's local bin directory to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"
# Add the non-root user's local site-packages to PYTHONPATH
ENV PYTHONPATH="/home/appuser/.local/lib/python3.12/site-packages"

# Set environment variables (including Timezone)
ENV PYTHONUNBUFFERED=1
# Set timezone to India Standard Time
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install runtime OS dependencies (like libpq for psycopg2)
# Add tzdata for timezone support
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 tzdata && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user to run the application
RUN addgroup --system appgroup && adduser --system --ingroup appgroup --no-create-home --shell /bin/false appuser

# Copy installed Python dependencies from the builder stage's user site
# Ensure the target directory exists and has correct permissions before copying
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code and startup script from builder stage
COPY --from=builder /app/app /app/app
# COPY --from=builder /app/alembic.ini . # Removed as Alembic is not used
# COPY --from=builder /app/alembic ./alembic # Removed as Alembic is not used
# Copy startup script to /app
COPY ./app/core/startup.sh /app/
RUN chmod +x /app/startup.sh # Make startup script executable

# Ensure the non-root user owns the application files AND their .local directory
RUN chown -R appuser:appgroup /app /home/appuser/.local # Give ownership to app user

# Switch to the non-root user
USER appuser

# Expose the port the application runs on internally
EXPOSE 80

# Command to run the application using the startup script
CMD ["/app/startup.sh"] 