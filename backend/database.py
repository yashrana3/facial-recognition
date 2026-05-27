import mysql.connector
from datetime import datetime
import json
import time

# Database Credentials
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'codenia',
    'password': 'Codenia@123',
    'use_pure': True  # Ensure pure Python implementation if C extensions fail
}

DB_NAME = 'edtech_vision'

def get_db_connection(db_name=None):
    """
    Establishes a connection to the MySQL database.
    If db_name is provided, it connects to that specific database.
    """
    config = DB_CONFIG.copy()
    if db_name:
        config['database'] = db_name
    
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def init_db():
    """
    Initializes the database and creates the necessary tables if they don't exist.
    """
    # Connect to MySQL server first to create DB if needed
    conn = get_db_connection()
    if conn is None:
        return
    
    cursor = conn.cursor()
    
    try:
        # Create Database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        print(f"Database '{DB_NAME}' checked/created.")
        
        # Connect to the specific database
        conn.database = DB_NAME
        
        # Table: frame_logs
        # Stores detailed analysis for every frame
        create_frame_logs_table = """
        CREATE TABLE IF NOT EXISTS frame_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(255),
            emotion VARCHAR(50),
            confidence FLOAT,
            gaze VARCHAR(50),
            blink_rate FLOAT,
            head_in_stable_state BOOLEAN,
            understanding_state VARCHAR(50),
            engagement_index FLOAT,
            raw_data JSON
        )
        """
        cursor.execute(create_frame_logs_table)
        print("Table 'frame_logs' checked/created.")
        
        # Table: hourly_reports
        # Stores aggregated stats
        create_hourly_reports_table = """
        CREATE TABLE IF NOT EXISTS hourly_reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            report_start_time DATETIME,
            report_end_time DATETIME,
            session_id VARCHAR(255),
            avg_engagement FLOAT,
            dominant_emotion VARCHAR(50),
            top_understanding_state VARCHAR(50),
            total_frames INT,
            notes TEXT
        )
        """
        cursor.execute(create_hourly_reports_table)
        print("Table 'hourly_reports' checked/created.")
        
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error initializing database: {err}")
    finally:
        cursor.close()
        conn.close()

def log_frame(data, session_id="default_session"):
    """
    Inserts a single frame analysis record into the frame_logs table.
    """
    conn = get_db_connection(DB_NAME)
    if conn is None:
        return
    
    cursor = conn.cursor()
    
    try:
        query = """
        INSERT INTO frame_logs 
        (session_id, emotion, confidence, gaze, blink_rate, head_in_stable_state, understanding_state, engagement_index, raw_data) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Prepare data
        raw_json = json.dumps(data)
        values = (
            session_id,
            data.get('emotion', 'unknown'),
            data.get('confidence', 0.0),
            data.get('gaze', 'unknown'),
            data.get('blink_rate', 0.0),
            data.get('head_stable', False),
            data.get('understanding', 'unknown'),
            data.get('engagement_index', 0.0),
            raw_json
        )
        
        cursor.execute(query, values)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error logging frame: {err}")
    finally:
        cursor.close()
        conn.close()

def aggregate_last_hour(session_id="default_session"):
    """
    Aggregates data for the last hour and stores it in hourly_reports.
    This is a simplified implementation.
    """
    conn = get_db_connection(DB_NAME)
    if conn is None:
        return
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get data from the last hour
        query = """
        SELECT * FROM frame_logs 
        WHERE session_id = %s 
        AND timestamp >= NOW() - INTERVAL 1 HOUR
        """
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        
        if not rows:
            print("No data to aggregate for the last hour.")
            return

        total_frames = len(rows)
        avg_engagement = sum(row['engagement_index'] for row in rows) / total_frames
        
        # Find dominant emotion
        emotions = [row['emotion'] for row in rows]
        most_common_emotion = max(set(emotions), key=emotions.count)
        
        # Find top understanding state
        states = [row['understanding_state'] for row in rows]
        most_common_state = max(set(states), key=states.count)
        
        # Insert report
        insert_query = """
        INSERT INTO hourly_reports 
        (report_start_time, report_end_time, session_id, avg_engagement, dominant_emotion, top_understanding_state, total_frames, notes)
        VALUES (NOW() - INTERVAL 1 HOUR, NOW(), %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            session_id, 
            avg_engagement, 
            most_common_emotion, 
            most_common_state, 
            total_frames, 
            "Auto-generated hourly report"
        ))
        
        conn.commit()
        print(f"Hourly report generated for session {session_id}.")
        
    except mysql.connector.Error as err:
        print(f"Error aggregating data: {err}")
    finally:
        cursor.close()
        conn.close()

def aggregate_all_active_sessions():
    """
    Finds all sessions active in the last hour and aggregates their data.
    """
    conn = get_db_connection(DB_NAME)
    if conn is None:
        return
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Find distinct sessions with activity in the last hour
        query = """
        SELECT DISTINCT session_id FROM frame_logs 
        WHERE timestamp >= NOW() - INTERVAL 1 HOUR
        """
        cursor.execute(query)
        sessions = cursor.fetchall()
        
        for session in sessions:
            session_id = session['session_id']
            aggregate_last_hour(session_id)
            
    except mysql.connector.Error as err:
        print(f"Error finding active sessions: {err}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_db()
