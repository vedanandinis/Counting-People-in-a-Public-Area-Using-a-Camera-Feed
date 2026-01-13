"""
JWT-Based Authentication System
Complete implementation with token generation, refresh, and verification
"""

from flask import Flask, request, jsonify, make_response
from functools import wraps
import jwt
import datetime
import hashlib
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['JWT_ALGORITHM'] = 'HS256'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = 2592000  # 30 days

# ==================== DATABASE ====================
class AuthDatabase:
    def __init__(self, db_path='auth.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize authentication database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table with hashed passwords
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                failed_attempts INTEGER DEFAULT 0,
                locked_until DATETIME
            )
        ''')
        
        # Refresh tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                revoked INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Login history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                success INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        
        # Create default admin user
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            password_hash = generate_password_hash('admin123')
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', ('admin', password_hash, 'admin@example.com', 'admin'))
            conn.commit()
            print("✅ Default admin user created: admin / admin123")
        
        conn.close()
    
    def get_user_by_username(self, username):
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        return user
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        return user
    
    def create_user(self, username, password, email, role='user'):
        """Create new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = generate_password_hash(password, method='sha256')
        
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, email, role))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return True, user_id, "User created successfully"
        except sqlite3.IntegrityError as e:
            conn.close()
            if 'username' in str(e):
                return False, None, "Username already exists"
            else:
                return False, None, "Email already exists"
    
    def verify_password(self, username, password):
        """Verify user password"""
        user = self.get_user_by_username(username)
        if not user:
            return False, None
        
        # Check if account is locked
        if user[9]:  # locked_until column
            locked_until = datetime.datetime.strptime(user[9], '%Y-%m-%d %H:%M:%S')
            if datetime.datetime.now() < locked_until:
                return False, "Account is locked due to multiple failed login attempts"
        
        # Check if account is active
        if not user[5]:  # is_active column
            return False, "Account is disabled"
        
        # Verify password
        if check_password_hash(user[2], password):  # password_hash column
            # Reset failed attempts
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET failed_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user[0],))
            conn.commit()
            conn.close()
            return True, user
        else:
            # Increment failed attempts
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            failed_attempts = user[8] + 1
            
            # Lock account after 5 failed attempts
            if failed_attempts >= 5:
                locked_until = datetime.datetime.now() + datetime.timedelta(minutes=30)
                cursor.execute('''
                    UPDATE users 
                    SET failed_attempts = ?, locked_until = ?
                    WHERE id = ?
                ''', (failed_attempts, locked_until, user[0]))
                conn.commit()
                conn.close()
                return False, "Account locked due to multiple failed attempts. Try again in 30 minutes."
            else:
                cursor.execute('''
                    UPDATE users SET failed_attempts = ? WHERE id = ?
                ''', (failed_attempts, user[0]))
                conn.commit()
                conn.close()
                return False, f"Invalid password. {5 - failed_attempts} attempts remaining."
    
    def save_refresh_token(self, user_id, token, expires_at):
        """Save refresh token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO refresh_tokens (user_id, token, expires_at)
            VALUES (?, ?, ?)
        ''', (user_id, token, expires_at))
        conn.commit()
        conn.close()
    
    def verify_refresh_token(self, token):
        """Verify refresh token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, expires_at, revoked 
            FROM refresh_tokens 
            WHERE token = ?
        ''', (token,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False, None
        
        user_id, expires_at, revoked = result
        
        if revoked:
            return False, None
        
        if datetime.datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S') < datetime.datetime.now():
            return False, None
        
        return True, user_id
    
    def revoke_refresh_token(self, token):
        """Revoke refresh token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE refresh_tokens SET revoked = 1 WHERE token = ?', (token,))
        conn.commit()
        conn.close()
    
    def log_login_attempt(self, user_id, ip_address, user_agent, success):
        """Log login attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO login_history (user_id, ip_address, user_agent, success)
            VALUES (?, ?, ?, ?)
        ''', (user_id, ip_address, user_agent, success))
        conn.commit()
        conn.close()

# Initialize database
auth_db = AuthDatabase()

# ==================== JWT FUNCTIONS ====================
def generate_access_token(user_id, username, role):
    """Generate JWT access token"""
    payload = {
        'user_id': user_id,
        'username': username,
        'role': role,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=app.config['JWT_ACCESS_TOKEN_EXPIRES']),
        'iat': datetime.datetime.utcnow(),
        'type': 'access'
    }
    
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm=app.config['JWT_ALGORITHM'])
    return token

def generate_refresh_token(user_id, username):
    """Generate JWT refresh token"""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=app.config['JWT_REFRESH_TOKEN_EXPIRES']),
        'iat': datetime.datetime.utcnow(),
        'type': 'refresh'
    }
    
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm=app.config['JWT_ALGORITHM'])
    return token

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=[app.config['JWT_ALGORITHM']])
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, 'Token has expired'
    except jwt.InvalidTokenError:
        return False, 'Invalid token'

# ==================== DECORATORS ====================
def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'success': False, 'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'success': False, 'message': 'Token is missing'}), 401
        
        # Verify token
        valid, payload = verify_token(token)
        
        if not valid:
            return jsonify({'success': False, 'message': payload}), 401
        
        # Check token type
        if payload.get('type') != 'access':
            return jsonify({'success': False, 'message': 'Invalid token type'}), 401
        
        # Add user info to request context
        request.current_user = payload
        
        return f(*args, **kwargs)
    
    return decorated

def role_required(allowed_roles):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'success': False, 'message': 'Authentication required'}), 401
            
            user_role = request.current_user.get('role')
            
            if user_role not in allowed_roles:
                return jsonify({'success': False, 'message': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        
        return decorated
    return decorator

# ==================== API ENDPOINTS ====================
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    role = data.get('role', 'user')
    
    if not username or not password or not email:
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Validate password strength
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400
    
    success, user_id, message = auth_db.create_user(username, password, email, role)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'user_id': user_id
        }), 201
    else:
        return jsonify({'success': False, 'message': message}), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login and get tokens"""
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing username or password'}), 400
    
    # Verify credentials
    valid, result = auth_db.verify_password(username, password)
    
    # Get IP and user agent
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    if not valid:
        # Log failed attempt
        user = auth_db.get_user_by_username(username)
        if user:
            auth_db.log_login_attempt(user[0], ip_address, user_agent, 0)
        
        return jsonify({'success': False, 'message': result}), 401
    
    # User authenticated successfully
    user = result
    user_id = user[0]
    user_role = user[4]
    
    # Generate tokens
    access_token = generate_access_token(user_id, username, user_role)
    refresh_token = generate_refresh_token(user_id, username)
    
    # Save refresh token
    expires_at = datetime.datetime.now() + datetime.timedelta(seconds=app.config['JWT_REFRESH_TOKEN_EXPIRES'])
    auth_db.save_refresh_token(user_id, refresh_token, expires_at)
    
    # Log successful login
    auth_db.log_login_attempt(user_id, ip_address, user_agent, 1)
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'token_type': 'Bearer',
        'expires_in': app.config['JWT_ACCESS_TOKEN_EXPIRES'],
        'user': {
            'id': user_id,
            'username': username,
            'email': user[3],
            'role': user_role
        }
    }), 200

@app.route('/api/auth/refresh', methods=['POST'])
def refresh():
    """Refresh access token using refresh token"""
    data = request.get_json()
    refresh_token = data.get('refresh_token')
    
    if not refresh_token:
        return jsonify({'success': False, 'message': 'Refresh token is missing'}), 400
    
    # Verify refresh token in database
    valid, user_id = auth_db.verify_refresh_token(refresh_token)
    
    if not valid:
        return jsonify({'success': False, 'message': 'Invalid or expired refresh token'}), 401
    
    # Get user info
    user = auth_db.get_user_by_id(user_id)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Generate new access token
    access_token = generate_access_token(user[0], user[1], user[4])
    
    return jsonify({
        'success': True,
        'access_token': access_token,
        'token_type': 'Bearer',
        'expires_in': app.config['JWT_ACCESS_TOKEN_EXPIRES']
    }), 200

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout():
    """Logout and revoke refresh token"""
    data = request.get_json()
    refresh_token = data.get('refresh_token')
    
    if refresh_token:
        auth_db.revoke_refresh_token(refresh_token)
    
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    }), 200

@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user():
    """Get current user info"""
    user_id = request.current_user['user_id']
    user = auth_db.get_user_by_id(user_id)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    return jsonify({
        'success': True,
        'user': {
            'id': user[0],
            'username': user[1],
            'email': user[3],
            'role': user[4],
            'created_at': user[6],
            'last_login': user[7]
        }
    }), 200

@app.route('/api/protected', methods=['GET'])
@token_required
def protected_route():
    """Example protected route"""
    return jsonify({
        'success': True,
        'message': 'This is a protected route',
        'user': request.current_user
    }), 200

@app.route('/api/admin-only', methods=['GET'])
@token_required
@role_required(['admin'])
def admin_only_route():
    """Example admin-only route"""
    return jsonify({
        'success': True,
        'message': 'This route is only accessible by admins',
        'user': request.current_user
    }), 200

# ==================== MAIN ====================
if __name__ == '__main__':
    print("="*60)
    print("🔐 JWT Authentication System Started")
    print("="*60)
    print("Default Login: admin / admin123")
    print("="*60)
    print("\nAPI Endpoints:")
    print("POST /api/auth/register   - Register new user")
    print("POST /api/auth/login      - Login and get tokens")
    print("POST /api/auth/refresh    - Refresh access token")
    print("POST /api/auth/logout     - Logout")
    print("GET  /api/auth/me         - Get current user info")
    print("GET  /api/protected       - Protected route example")
    print("GET  /api/admin-only      - Admin-only route example")
    print("="*60)
    app.run(host='0.0.0.0', port=5002, debug=True)