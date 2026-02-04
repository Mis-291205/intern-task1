from locust import HttpUser, task, between

class PredictionUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_temperature(self):
        self.client.post(
            "/predict",
            json={
                "meantemp": 25,
                "humidity": 70,
                "wind_speed": 3,
                "meanpressure": 1012
            }
        )
