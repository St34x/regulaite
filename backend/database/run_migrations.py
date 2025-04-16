"""
Script to run database migrations on application startup.
"""
import os
import logging
import mariadb
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get a connection to the MariaDB database."""
    # Get database configuration from environment variables
    config = {
        'host': os.getenv('MARIADB_HOST', 'mariadb'),
        'port': int(os.getenv('MARIADB_PORT', 3306)),
        'user': os.getenv('MARIADB_USER', 'root'),
        'password': os.getenv('MARIADB_PASSWORD', 'password'),
        'database': os.getenv('MARIADB_DATABASE', 'regulaite')
    }
    
    # Try to connect to the database
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = mariadb.connect(**config)
            logger.info("Successfully connected to MariaDB database")
            return conn
        except mariadb.Error as e:
            logger.warning(f"Error connecting to MariaDB database (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Wait before retrying
                time.sleep(5)
    
    # If we get here, all connection attempts failed
    logger.error("Failed to connect to MariaDB database after multiple attempts")
    raise Exception("Could not connect to database")

def execute_migration_file(cursor, file_path: str) -> bool:
    """Execute a single migration file."""
    try:
        logger.info(f"Executing migration: {os.path.basename(file_path)}")
        
        with open(file_path, 'r') as f:
            sql_script = f.read()
            
        # Split the script into individual statements
        statements = sql_script.split(';')
        
        # Execute each statement (except empty ones)
        for statement in statements:
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
                
        logger.info(f"Migration executed successfully: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        logger.error(f"Error executing migration {os.path.basename(file_path)}: {str(e)}")
        return False

def get_applied_migrations(cursor) -> list:
    """Get a list of migrations that have already been applied."""
    try:
        # Create the migrations table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (name)
            )
        """)
        
        # Get already applied migrations
        cursor.execute("SELECT name FROM migrations ORDER BY id")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error checking applied migrations: {str(e)}")
        return []

def run_migrations(migration_dir: Optional[str] = None) -> bool:
    """Run all pending database migrations."""
    # Use the migrations directory in the same directory as this script if not specified
    if migration_dir is None:
        migration_dir = os.path.join(os.path.dirname(__file__), 'migrations')
    
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get already applied migrations
        applied_migrations = get_applied_migrations(cursor)
        logger.info(f"Found {len(applied_migrations)} applied migrations")
        
        # Get all migration files
        migration_files = []
        for filename in os.listdir(migration_dir):
            if filename.endswith('.sql'):
                migration_files.append(filename)
        
        # Sort migration files to ensure they're applied in the correct order
        migration_files.sort()
        
        # Apply all pending migrations
        success = True
        for filename in migration_files:
            if filename not in applied_migrations:
                file_path = os.path.join(migration_dir, filename)
                
                # Execute the migration
                if execute_migration_file(cursor, file_path):
                    # Record the migration as applied
                    try:
                        cursor.execute(
                            "INSERT INTO migrations (name) VALUES (?)",
                            (filename,)
                        )
                        conn.commit()
                        logger.info(f"Recorded migration as applied: {filename}")
                    except Exception as e:
                        logger.error(f"Error recording migration as applied: {str(e)}")
                        success = False
                else:
                    success = False
        
        # Close the database connection
        conn.close()
        
        if success:
            logger.info("All database migrations applied successfully")
        else:
            logger.warning("Some migrations failed to apply")
        
        return success
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        return False

if __name__ == "__main__":
    # Run migrations when the script is executed directly
    run_migrations() 