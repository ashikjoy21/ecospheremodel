import requests
import json
import os
from debug_utils import log_error, log_info

class RoboflowClient:
    def __init__(self, api_key, project_id, model_version):
        self.api_key = api_key
        self.project_id = project_id
        self.model_version = model_version
        self.base_url = f"https://detect.roboflow.com/{project_id}/{model_version}"
        log_info(f"Initialized custom Roboflow client for project {project_id}, version {model_version}")
        
    def predict(self, image_bytes, confidence=0.5):
        try:
            log_info(f"Sending prediction request with confidence {confidence}")
            
            # Use files parameter to properly send image data
            files = {
                "file": ("image.jpg", image_bytes, "image/jpeg")
            }
            
            params = {
                "api_key": self.api_key,
                "confidence": confidence,
                "format": "json"
            }
            
            # Don't set content-type header - let requests handle it for multipart/form-data
            response = requests.post(
                self.base_url, 
                params=params,
                files=files
            )
            
            if response.status_code != 200:
                log_error(f"Roboflow API error: {response.status_code} - {response.text}")
                raise Exception(f"API returned status code {response.status_code}: {response.text}")
                
            # Create response object with json method
            class ResponseWrapper:
                def __init__(self, data):
                    self.data = data
                    
                def json(self):
                    return self.data
            
            result = response.json()
            log_info(f"Received response with {len(result.get('predictions', []))} predictions")
            return ResponseWrapper(result)
            
        except Exception as e:
            log_error(f"Prediction request failed: {str(e)}")
            raise 