#!/bin/bash
set -e

# Use env vars or defaults
PGUSER=${PGUSER:-julia}
PGDATABASE=${PGDATABASE:-msft_customers}

# Directory holding the CSV files. No default: callers must set this
# explicitly (script/pg_ephemeral.sh, script/init.sh, init-docker-compose.sh).
: "${AAS_DATA_DIR:?AAS_DATA_DIR must be set (directory containing the CSV files)}"

echo "Using user: $PGUSER"
echo "Using database: $PGDATABASE"

echo "Starting import of CSV files..."

import_csv() {
  local table=$1
  local file=$2
  echo "Importing $file into table $table..."
  psql -U "$PGUSER" -d "$PGDATABASE" -c "\COPY $table FROM '$file' CSV HEADER NULL '';"
}

# Import tables in dependency order
import_csv ProductModel ${AAS_DATA_DIR}/ProductModel.csv
import_csv ProductCategory ${AAS_DATA_DIR}/ProductCategory.csv
import_csv ProductDescription ${AAS_DATA_DIR}/ProductDescription.csv
import_csv Product ${AAS_DATA_DIR}/Product.csv
import_csv Customer ${AAS_DATA_DIR}/Customer.csv
import_csv Address ${AAS_DATA_DIR}/Address.csv
import_csv CustomerAddress ${AAS_DATA_DIR}/CustomerAddress.csv
import_csv SalesOrderHeader ${AAS_DATA_DIR}/SalesOrderHeader.csv
import_csv SalesOrderDetail ${AAS_DATA_DIR}/SalesOrderDetail.csv

echo "CSV import complete."
