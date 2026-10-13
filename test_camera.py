"""Webcam test for SignBridge AI (Phase 1).

Opens the default webcam, shows the live feed, and exits when Q is pressed.
"""

import cv2


def main():
    # Open the default webcam (device index 0)
    camera = cv2.VideoCapture(0)

    # Check whether the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Make sure a camera is connected and not being used by another application.")
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
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
