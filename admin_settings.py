"""
Admin Settings System
Manage alert rules, and system configuration
"""

from flask import Flask, request, jsonify
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)

# ==================== DATABASE ====================
class SettingsDatabase:
    def __init__(self, db_path='settings.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize settings database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # System settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT,
                setting_type TEXT,
                category TEXT,
                description TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        ''')
        
        # Alert rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT UNIQUE NOT NULL,
                rule_type TEXT NOT NULL,
                conditions TEXT NOT NULL,
                actions TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                is_active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                updated_at DATETIME,
                updated_by TEXT
            )
        ''')
        
        # Zone thresholds table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zone_thresholds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                capacity INTEGER NOT NULL,
                warning_threshold REAL NOT NULL,
                critical_threshold REAL NOT NULL,
                alert_cooldown INTEGER DEFAULT 300,
                is_active INTEGER DEFAULT 1,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        ''')
        
        # Notification settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notification_type TEXT NOT NULL,
                is_enabled INTEGER DEFAULT 1,
                recipients TEXT,
                config TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        ''')
        
        # Settings history (audit trail)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                setting_type TEXT,
                setting_id INTEGER,
                old_value TEXT,
                new_value TEXT,
                changed_by TEXT,
                change_reason TEXT
            )
        ''')
        
        conn.commit()
        
        # Insert default settings
        self._insert_default_settings(cursor)
        conn.commit()
        conn.close()
        
        print("✅ Settings Database initialized")
    
    def _insert_default_settings(self, cursor):
        """Insert default system settings"""
        default_settings = [
            ('detection_confidence_threshold', '0.5', 'float', 'detection', 'Minimum confidence for person detection'),
            ('detection_fps', '10', 'integer', 'detection', 'Frames per second for detection'),
            ('max_people_count', '1000', 'integer', 'detection', 'Maximum people count per zone'),
            ('alert_cooldown_seconds', '300', 'integer', 'alerts', 'Seconds between duplicate alerts'),
            ('enable_email_alerts', 'true', 'boolean', 'alerts', 'Enable email notifications'),
            ('enable_sms_alerts', 'false', 'boolean', 'alerts', 'Enable SMS notifications'),
            ('enable_webhook_alerts', 'false', 'boolean', 'alerts', 'Enable webhook notifications'),
            ('data_retention_days', '90', 'integer', 'database', 'Days to retain detection data'),
            ('auto_cleanup_enabled', 'true', 'boolean', 'database', 'Automatically cleanup old data'),
            ('system_timezone', 'UTC', 'string', 'system', 'System timezone'),
            ('max_concurrent_cameras', '10', 'integer', 'system', 'Maximum concurrent camera streams')
        ]
        
        for setting in default_settings:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO system_settings 
                    (setting_key, setting_value, setting_type, category, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', setting)
            except:
                pass
        
        # Insert default alert rules
        default_rules = [
            (
                'capacity_warning',
                'threshold',
                json.dumps({'condition': 'capacity >= warning_threshold', 'metric': 'occupancy'}),
                json.dumps({'email': True, 'sms': False, 'webhook': False}),
                'medium'
            ),
            (
                'capacity_critical',
                'threshold',
                json.dumps({'condition': 'capacity >= critical_threshold', 'metric': 'occupancy'}),
                json.dumps({'email': True, 'sms': True, 'webhook': True}),
                'high'
            ),
            (
                'sudden_crowd_increase',
                'rate_of_change',
                json.dumps({'condition': 'increase > 50% in 5 minutes', 'metric': 'people_count'}),
                json.dumps({'email': True, 'sms': False, 'webhook': True}),
                'medium'
            )
        ]
        
        for rule in default_rules:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO alert_rules
                    (rule_name, rule_type, conditions, actions, priority)
                    VALUES (?, ?, ?, ?, ?)
                ''', rule)
            except:
                pass
    
    # ==================== SYSTEM SETTINGS ====================
    def get_all_settings(self, category=None):
        """Get all system settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('''
                SELECT * FROM system_settings WHERE category = ? ORDER BY setting_key
            ''', (category,))
        else:
            cursor.execute('SELECT * FROM system_settings ORDER BY category, setting_key')
        
        settings = cursor.fetchall()
        conn.close()
        return settings
    
    def get_setting(self, setting_key):
        """Get single setting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM system_settings WHERE setting_key = ?', (setting_key,))
        setting = cursor.fetchone()
        conn.close()
        return setting
    
    def update_setting(self, setting_key, new_value, username, reason=''):
        """Update system setting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get old value
        cursor.execute('SELECT setting_value FROM system_settings WHERE setting_key = ?', (setting_key,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False, "Setting not found"
        
        old_value = result[0]
        
        # Update setting
        cursor.execute('''
            UPDATE system_settings
            SET setting_value = ?, updated_at = CURRENT_TIMESTAMP, updated_by = ?
            WHERE setting_key = ?
        ''', (new_value, username, setting_key))
        
        # Log change
        cursor.execute('''
            INSERT INTO settings_history
            (setting_type, setting_id, old_value, new_value, changed_by, change_reason)
            VALUES (?, (SELECT id FROM system_settings WHERE setting_key = ?), ?, ?, ?, ?)
        ''', ('system_setting', setting_key, old_value, new_value, username, reason))
        
        conn.commit()
        conn.close()
        
        return True, "Setting updated successfully"
    
    def get_settings_by_category(self):
        """Get settings grouped by category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT category FROM system_settings ORDER BY category')
        categories = [row[0] for row in cursor.fetchall()]
        
        result = {}
        for category in categories:
            cursor.execute('''
                SELECT setting_key, setting_value, setting_type, description
                FROM system_settings
                WHERE category = ?
                ORDER BY setting_key
            ''', (category,))
            
            result[category] = [dict(zip(['key', 'value', 'type', 'description'], row)) 
                              for row in cursor.fetchall()]
        
        conn.close()
        return result
    
    # ==================== ALERT RULES ====================
    def get_all_alert_rules(self):
        """Get all alert rules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM alert_rules ORDER BY priority DESC, rule_name')
        rules = cursor.fetchall()
        conn.close()
        return rules
    
    def get_alert_rule(self, rule_id):
        """Get single alert rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM alert_rules WHERE id = ?', (rule_id,))
        rule = cursor.fetchone()
        conn.close()
        return rule
    
    def create_alert_rule(self, rule_data, username):
        """Create new alert rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO alert_rules
                (rule_name, rule_type, conditions, actions, priority, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                rule_data['rule_name'],
                rule_data['rule_type'],
                json.dumps(rule_data['conditions']),
                json.dumps(rule_data['actions']),
                rule_data.get('priority', 'medium'),
                username
            ))
            
            conn.commit()
            rule_id = cursor.lastrowid
            conn.close()
            
            return True, rule_id, "Alert rule created successfully"
        
        except sqlite3.IntegrityError:
            conn.close()
            return False, None, "Rule name already exists"
        except Exception as e:
            conn.close()
            return False, None, str(e)
    
    def update_alert_rule(self, rule_id, rule_data, username):
        """Update alert rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get old values
        cursor.execute('SELECT * FROM alert_rules WHERE id = ?', (rule_id,))
        old_rule = cursor.fetchone()
        
        if not old_rule:
            conn.close()
            return False, "Rule not found"
        
        try:
            cursor.execute('''
                UPDATE alert_rules
                SET rule_name = ?, rule_type = ?, conditions = ?, actions = ?,
                    priority = ?, updated_at = CURRENT_TIMESTAMP, updated_by = ?
                WHERE id = ?
            ''', (
                rule_data.get('rule_name'),
                rule_data.get('rule_type'),
                json.dumps(rule_data.get('conditions')),
                json.dumps(rule_data.get('actions')),
                rule_data.get('priority'),
                username,
                rule_id
            ))
            
            # Log change
            cursor.execute('''
                INSERT INTO settings_history
                (setting_type, setting_id, old_value, new_value, changed_by)
                VALUES (?, ?, ?, ?, ?)
            ''', ('alert_rule', rule_id, json.dumps(old_rule), json.dumps(rule_data), username))
            
            conn.commit()
            conn.close()
            
            return True, "Alert rule updated successfully"
        
        except Exception as e:
            conn.close()
            return False, str(e)
    
    def delete_alert_rule(self, rule_id, username):
        """Delete alert rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM alert_rules WHERE id = ?', (rule_id,))
        
        # Log change
        cursor.execute('''
            INSERT INTO settings_history
            (setting_type, setting_id, changed_by, change_reason)
            VALUES (?, ?, ?, ?)
        ''', ('alert_rule', rule_id, username, 'Rule deleted'))
        
        conn.commit()
        conn.close()
        
        return True, "Alert rule deleted successfully"
    
    def toggle_alert_rule(self, rule_id, username):
        """Toggle alert rule active status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT is_active FROM alert_rules WHERE id = ?', (rule_id,))
        current_status = cursor.fetchone()[0]
        new_status = 0 if current_status else 1
        
        cursor.execute('''
            UPDATE alert_rules
            SET is_active = ?, updated_at = CURRENT_TIMESTAMP, updated_by = ?
            WHERE id = ?
        ''', (new_status, username, rule_id))
        
        conn.commit()
        conn.close()
        
        return True, f"Rule {'activated' if new_status else 'deactivated'}"
    
    # ==================== ZONE THRESHOLDS ====================
    def get_zone_thresholds(self, zone_id=None):
        """Get zone thresholds"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if zone_id:
            cursor.execute('SELECT * FROM zone_thresholds WHERE zone_id = ?', (zone_id,))
            threshold = cursor.fetchone()
        else:
            cursor.execute('SELECT * FROM zone_thresholds')
            threshold = cursor.fetchall()
        
        conn.close()
        return threshold
    
    def upsert_zone_threshold(self, zone_id, threshold_data, username):
        """Update or insert zone threshold"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute('SELECT id FROM zone_thresholds WHERE zone_id = ?', (zone_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update
            cursor.execute('''
                UPDATE zone_thresholds
                SET capacity = ?, warning_threshold = ?, critical_threshold = ?,
                    alert_cooldown = ?, updated_at = CURRENT_TIMESTAMP, updated_by = ?
                WHERE zone_id = ?
            ''', (
                threshold_data['capacity'],
                threshold_data['warning_threshold'],
                threshold_data['critical_threshold'],
                threshold_data.get('alert_cooldown', 300),
                username,
                zone_id
            ))
        else:
            # Insert
            cursor.execute('''
                INSERT INTO zone_thresholds
                (zone_id, capacity, warning_threshold, critical_threshold, alert_cooldown, updated_by)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                zone_id,
                threshold_data['capacity'],
                threshold_data['warning_threshold'],
                threshold_data['critical_threshold'],
                threshold_data.get('alert_cooldown', 300),
                username
            ))
        
        conn.commit()
        conn.close()
        
        return True, "Zone threshold updated successfully"
    
    # ==================== NOTIFICATION SETTINGS ====================
    def get_notification_settings(self):
        """Get all notification settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM notification_settings')
        settings = cursor.fetchall()
        conn.close()
        return settings
    
    def update_notification_setting(self, notification_type, is_enabled, recipients, config, username):
        """Update notification setting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute('SELECT id FROM notification_settings WHERE notification_type = ?', (notification_type,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE notification_settings
                SET is_enabled = ?, recipients = ?, config = ?,
                    updated_at = CURRENT_TIMESTAMP, updated_by = ?
                WHERE notification_type = ?
            ''', (is_enabled, recipients, config, username, notification_type))
        else:
            cursor.execute('''
                INSERT INTO notification_settings
                (notification_type, is_enabled, recipients, config, updated_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (notification_type, is_enabled, recipients, config, username))
        
        conn.commit()
        conn.close()
        
        return True, "Notification setting updated successfully"
    
    # ==================== SETTINGS HISTORY ====================
    def get_settings_history(self, limit=50):
        """Get settings change history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM settings_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        history = cursor.fetchall()
        conn.close()
        return history

db = SettingsDatabase()

# ==================== API ENDPOINTS ====================

# System Settings
@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get all system settings"""
    category = request.args.get('category')
    
    if category:
        settings = db.get_all_settings(category)
    else:
        settings = db.get_all_settings()
    
    return jsonify({
        'success': True,
        'data': [dict(zip(
            ['id', 'key', 'value', 'type', 'category', 'description', 'updated_at', 'updated_by'],
            setting
        )) for setting in settings]
    }), 200

@app.route('/api/settings/grouped', methods=['GET'])
def get_settings_grouped():
    """Get settings grouped by category"""
    settings = db.get_settings_by_category()
    return jsonify({'success': True, 'data': settings}), 200

@app.route('/api/settings/<setting_key>', methods=['PUT'])
def update_setting(setting_key):
    """Update system setting"""
    data = request.get_json()
    username = request.headers.get('X-Username', 'unknown')
    
    success, message = db.update_setting(
        setting_key,
        data['value'],
        username,
        data.get('reason', '')
    )
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    return jsonify({'success': False, 'message': message}), 400

# Alert Rules
@app.route('/api/alert-rules', methods=['GET'])
def get_alert_rules():
    """Get all alert rules"""
    rules = db.get_all_alert_rules()
    return jsonify({
        'success': True,
        'data': [dict(zip(
            ['id', 'rule_name', 'rule_type', 'conditions', 'actions', 'priority',
             'is_active', 'created_at', 'created_by', 'updated_at', 'updated_by'],
            rule
        )) for rule in rules]
    }), 200

@app.route('/api/alert-rules', methods=['POST'])
def create_alert_rule():
    """Create new alert rule"""
    data = request.get_json()
    username = request.headers.get('X-Username', 'unknown')
    
    success, rule_id, message = db.create_alert_rule(data, username)
    
    if success:
        return jsonify({'success': True, 'message': message, 'rule_id': rule_id}), 201
    return jsonify({'success': False, 'message': message}), 400

@app.route('/api/alert-rules/<int:rule_id>', methods=['PUT'])
def update_alert_rule(rule_id):
    """Update alert rule"""
    data = request.get_json()
    username = request.headers.get('X-Username', 'unknown')
    
    success, message = db.update_alert_rule(rule_id, data, username)
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    return jsonify({'success': False, 'message': message}), 400

@app.route('/api/alert-rules/<int:rule_id>', methods=['DELETE'])
def delete_alert_rule(rule_id):
    """Delete alert rule"""
    username = request.headers.get('X-Username', 'unknown')
    
    success, message = db.delete_alert_rule(rule_id, username)
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    return jsonify({'success': False, 'message': message}), 400

@app.route('/api/alert-rules/<int:rule_id>/toggle', methods=['POST'])
def toggle_alert_rule(rule_id):
    """Toggle alert rule"""
    username = request.headers.get('X-Username', 'unknown')
    
    success, message = db.toggle_alert_rule(rule_id, username)
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    return jsonify({'success': False, 'message': message}), 400

# Zone Thresholds
@app.route('/api/zone-thresholds', methods=['GET'])
def get_zone_thresholds():
    """Get zone thresholds"""
    zone_id = request.args.get('zone_id')
    
    thresholds = db.get_zone_thresholds(zone_id)
    
    if zone_id:
        if thresholds:
            return jsonify({
                'success': True,
                'data': dict(zip(
                    ['id', 'zone_id', 'capacity', 'warning_threshold', 'critical_threshold',
                     'alert_cooldown', 'is_active', 'updated_at', 'updated_by'],
                    thresholds
                ))
            }), 200
        return jsonify({'success': False, 'message': 'Threshold not found'}), 404
    else:
        return jsonify({
            'success': True,
            'data': [dict(zip(
                ['id', 'zone_id', 'capacity', 'warning_threshold', 'critical_threshold',
                 'alert_cooldown', 'is_active', 'updated_at', 'updated_by'],
                threshold
            )) for threshold in thresholds]
        }), 200

@app.route('/api/zone-thresholds/<zone_id>', methods=['POST', 'PUT'])
def upsert_zone_threshold(zone_id):
    """Create or update zone threshold"""
    data = request.get_json()
    username = request.headers.get('X-Username', 'unknown')
    
    success, message = db.upsert_zone_threshold(zone_id, data, username)
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    return jsonify({'success': False, 'message': message}), 400

# Notification Settings
@app.route('/api/notification-settings', methods=['GET'])
def get_notification_settings():
    """Get notification settings"""
    settings = db.get_notification_settings()
    return jsonify({
        'success': True,
        'data': [dict(zip(
            ['id', 'notification_type', 'is_enabled', 'recipients', 'config', 'updated_at', 'updated_by'],
            setting
        )) for setting in settings]
    }), 200

@app.route('/api/notification-settings/<notification_type>', methods=['POST', 'PUT'])
def update_notification_setting(notification_type):
    """Update notification setting"""
    data = request.get_json()
    username = request.headers.get('X-Username', 'unknown')
    
    success, message = db.update_notification_setting(
        notification_type,
        data['is_enabled'],
        data.get('recipients'),
        data.get('config'),
        username
    )
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    return jsonify({'success': False, 'message': message}), 400

# Settings History
@app.route('/api/settings/history', methods=['GET'])
def get_settings_history():
    """Get settings change history"""
    limit = int(request.args.get('limit', 50))
    
    history = db.get_settings_history(limit)
    
    return jsonify({
        'success': True,
        'data': [dict(zip(
            ['id', 'timestamp', 'setting_type', 'setting_id', 'old_value',
             'new_value', 'changed_by', 'change_reason'],
            entry
        )) for entry in history]
    }), 200

if __name__ == '__main__':
    print("="*60)
    print("⚙️  Admin Settings System")
    print("="*60)
    print("Running on http://localhost:5005")
    print("="*60)
    print("\nEndpoints:")
    print("GET  /api/settings                    - Get all settings")
    print("GET  /api/settings/grouped            - Get grouped settings")
    print("PUT  /api/settings/<key>              - Update setting")
    print("GET  /api/alert-rules                 - Get alert rules")
    print("POST /api/alert-rules                 - Create alert rule")
    print("PUT  /api/alert-rules/<id>            - Update alert rule")
    print("GET  /api/zone-thresholds             - Get zone thresholds")
    print("POST /api/zone-thresholds/<zone_id>   - Set zone threshold")
    print("GET  /api/notification-settings       - Get notification settings")
    print("GET  /api/settings/history            - Get change history")
    print("="*60)
    app.run(host='0.0.0.0', port=5005, debug=True)