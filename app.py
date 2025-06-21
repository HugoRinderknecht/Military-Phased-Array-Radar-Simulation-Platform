from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from config.settings import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
