def test_predict(client):
    payload = {
        "meantemp": 25,
        "humidity": 70,
        "wind_speed": 3,
        "meanpressure": 1012
    }

    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "predicted_next_day_temperature" in res.json()
