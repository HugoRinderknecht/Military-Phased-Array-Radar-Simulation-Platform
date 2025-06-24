# api/__init__.py
from .signal_routes import signal_bp
from .rcs_routes import rcs_bp
from .computation_routes import computation_bp

# 注册到主应用
def register_blueprints(app):
    app.register_blueprint(signal_bp, url_prefix='/api/signal')
    app.register_blueprint(rcs_bp, url_prefix='/api/rcs')
    app.register_blueprint(computation_bp, url_prefix='/api/computation')
