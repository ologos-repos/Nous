#!/bin/bash
# setup-postgres.sh — Create the Nous database and user in PostgreSQL.
#
# Usage:
#   bash scripts/setup-postgres.sh                     # defaults (nous/nous on localhost)
#   NOUS_DB=mydb NOUS_USER=myuser bash scripts/setup-postgres.sh  # custom
#
# Prerequisites:
#   - PostgreSQL server running
#   - psql CLI available
#   - Current user has CREATE DATABASE privileges (or run as postgres superuser)

set -euo pipefail

DB_NAME="${NOUS_DB:-nous}"
DB_USER="${NOUS_USER:-nous}"
DB_PASS="${NOUS_PASS:-nous}"
DB_HOST="${NOUS_HOST:-localhost}"
DB_PORT="${NOUS_PORT:-5432}"

echo "╔══════════════════════════════════════╗"
echo "║   Nous — PostgreSQL Database Setup   ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Database: ${DB_NAME}"
echo "User:     ${DB_USER}"
echo "Host:     ${DB_HOST}:${DB_PORT}"
echo ""

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo "✗ psql not found. Install PostgreSQL client tools first."
    exit 1
fi

# Create user (ignore error if exists)
echo "Creating user '${DB_USER}'..."
psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -c \
    "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';" 2>/dev/null || \
    echo "  (user already exists — skipping)"

# Create database (ignore error if exists)
echo "Creating database '${DB_NAME}'..."
psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -c \
    "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};" 2>/dev/null || \
    echo "  (database already exists — skipping)"

# Grant privileges
echo "Granting privileges..."
psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -c \
    "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"

echo ""
echo "✓ PostgreSQL setup complete."
echo ""
echo "Connection string:"
echo "  postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo ""
echo "Next steps:"
echo "  1. Run 'nous-setup init' to create tables and configure Nous"
echo "  2. Or use MemoryStore.connect() in your code with the connection string above"
