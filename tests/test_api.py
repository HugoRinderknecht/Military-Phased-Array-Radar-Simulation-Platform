import pytest
import json
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_simulation_request():
    return {
        "radar": {
            "radar_area": 100.0,
            "tr_components": 1000,
            "radar_power": 50000,
            "frequency": 10e9
        },
        "environment": {
            "weather_type": "clear",
            "clutter_density": 0.3,
            "interference_level": 0.1
        },
        "targets": {
            "num_targets": 5,
            "specific_targets": [
                {
                    "position": [50000, 30000, 5000],
                    "velocity": [300, 200, 0],
                    "rcs": 10.0,
                    "altitude": 5000,
                    "aspect_angle": 1.57
                }
            ]
        },
        "simulation_time": 5.0,
        "monte_carlo_runs": 2
    }


def test_health_check(client):
    response = client.get('/api/health')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'version' in data


def test_simulation_status(client):
    response = client.get('/api/status')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'data' in data


def test_run_simulation(client, sample_simulation_request):
    response = client.post('/api/simulate',
                           data=json.dumps(sample_simulation_request),
                           content_type='application/json')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'results' in data


def test_invalid_simulation_request(client):
    invalid_request = {
        "radar": {
            "radar_area": -1  # Invalid negative value
        }
    }

    response = client.post('/api/simulate',
                           data=json.dumps(invalid_request),
                           content_type='application/json')

    assert response.status_code == 400

    data = json.loads(response.data)
    assert data['status'] == 'error'


def test_analyze_results(client):
    sample_results = {
        "results": {
            "summary": {
                "total_runs": 1,
                "avg_final_tracks": 3,
                "avg_total_detections": 15
            },
            "time_series": {
                "0": {
                    "time": 0.0,
                    "avg_detections": 3,
                    "avg_tracks": 2,
                    "avg_confirmed_tracks": 1,
                    "avg_scheduling_efficiency": 0.8
                }
            }
        }
    }

    response = client.post('/api/analyze',
                           data=json.dumps(sample_results),
                           content_type='application/json')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'analysis' in data


def test_reset_simulation(client):
    response = client.post('/api/reset')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
