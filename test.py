# Import unittest module for writing unit tests and integration tests
import unittest

# Import app module from app.py file
from app import app

# Import os module for accessing file system
import os

# Create a TestApp class that inherits from unittest.TestCase class
class TestApp(unittest.TestCase):

    # Define a setUp method that runs before each test case
    def setUp(self):
        # Create a test client for the app using Flask's test_client method
        self.client = app.test_client()
        # Set the app testing mode to True using Flask's testing attribute
        app.testing = True
    
    # Define a tearDown method that runs after each test case
    def tearDown(self):
        # Delete all the files in the upload folder using os module's remove method
        for filename in os.listdir(app.config["UPLOADED_MEDIA_DEST"]):
            filepath = os.path.join(app.config["UPLOADED_MEDIA_DEST"], filename)
            os.remove(filepath)

    # Define a test_index method that tests the index route of the app using unittest's assert methods
    def test_index(self):
        # Send a GET request to the index route using test client's get method and get the response object
        response = self.client.get("/")
        # Check if the response status code is 200 (OK) using unittest's assertEqual method
        self.assertEqual(response.status_code, 200)
        # Check if the response content type is text/html using unittest's assertEqual method
        self.assertEqual(response.content_type, "text/html; charset=utf-8")
        # Check if the response data contains the title of the app using unittest's assertIn method
        self.assertIn(b"Fake AI Content Detection Program", response.data)

    # Define a test_detect method that tests the detect route of the app using unittest's assert methods
    def test_detect(self):
        # Define a list of media types and media files for testing
        media_types = ["video", "image", "text", "voice", "music"]
        media_files = ["test_video.mp4", "test_image.jpg", "test_text.txt", "test_voice.wav", "test_music.mp3"]
        # Loop over the media types and media files
        for media_type, media_file in zip(media_types, media_files):
            # Open the media file in binary mode using Python's built-in open function
            with open(media_file, "rb") as f:
                # Send a POST request to the detect route with the media type and the media file as form data using test client's post method and get the response object
                response = self.client.post("/detect", data={"media-type": media_type, "media-file": f})
                # Check if the response status code is 200 (OK) using unittest's assertEqual method
                self.assertEqual(response.status_code, 200)
                # Check if the response content type is application/json using unittest's assertEqual method
                self.assertEqual(response.content_type, "application/json")
                # Parse the response data as a JSON object using Flask's get_json method
                data = response.get_json()
                # Check if the data contains a probability key and its value is a float between 0 and 1 using unittest's assertIn and assertIsInstance methods
                self.assertIn("probability", data)
                self.assertIsInstance(data["probability"], float)
                self.assertGreaterEqual(data["probability"], 0)
                self.assertLessEqual(data["probability"], 1)
                # Check if the data contains a type key and its value is either None or a string that starts with Fake using unittest's assertIn and assertIsInstance methods
                self.assertIn("type", data)
                self.assertIsInstance(data["type"], (type(None), str))
                if data["type"] is not None:
                    self.assertTrue(data["type"].startswith("Fake"))
                # Check if the data contains an explanation key and its value is either None or a string that contains a model name using unittest's assertIn and assertIsInstance methods
                self.assertIn("explanation", data)
                self.assertIsInstance(data["explanation"], (type(None), str))
                if data["explanation"] is not None:
                    self.assertIn(data["type"].split()[1], data["explanation"])

# Run the tests using unittest's main function
if __name__ == "__main__":
    unittest.main()
