"""Webcam test for SignBridge AI (Phase 1).

Opens the default webcam, shows the live feed, and exits when Q is pressed.
"""

import sys
from pathlib import Path

import cv2

# Make the src/ package importable when this script is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import camera_lock


def main():
    # Advisory ownership (src/camera_lock.py): if the web app or another
    # SignBridge script already owns the camera, fail fast with its PID.
    try:
        owned = camera_lock.acquire(0)
    except camera_lock.CameraBusyError as busy:
        print(f"Error: webcam 0 is already owned by another SignBridge "
              f"process (PID {busy.holder_pid}) - stop it first.")
        return

    # Open the default webcam (device index 0)
    camera = cv2.VideoCapture(0)

    # Check whether the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Make sure a camera is connected and not being used by another application.")
        owned.release()
        return

    print("Webcam started. Press Q in the video window to quit.")

    try:
        # Read and display frames continuously
        while True:
            frame_ok, frame = camera.read()

            # If a frame could not be read, the camera was disconnected or failed
            if not frame_ok:
                print("Error: Failed to read a frame from the webcam.")
                break

            cv2.imshow("SignBridge AI - Camera Test", frame)

            # Wait 1 ms for a key press; quit when Q is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        # Allows closing with Ctrl+C in the terminal as well
        print("\nStopped by user.")

    finally:
        # Always release the camera and close the windows
        camera.release()
        owned.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
