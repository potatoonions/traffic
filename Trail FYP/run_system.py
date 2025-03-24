import subprocess
import time
import os
import sys

def start_server():
    """
    Start the FastAPI server in a new process
    """
    try:
        server_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--reload"],
                                        cwd=os.path.dirname(os.path.abspath(__file__)))
        print("Starting FastAPI server...")
        return server_process
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)

def start_gui():
    """
    Start the GUI application in a new process
    """
    try:
        gui_process = subprocess.Popen([sys.executable, "stock_gui.py"],
                                     cwd=os.path.dirname(os.path.abspath(__file__)))
        print("Starting GUI application...")
        return gui_process
    except Exception as e:
        print(f"Error starting GUI: {str(e)}")
        sys.exit(1)

def main():
    """
    Main function to start both server and GUI
    """
    print("Starting Stock Market Prediction System...")
    
    # Start server
    server_process = start_server()
    
    # Wait a few seconds for the server to start
    print("Waiting for server to initialize...")
    time.sleep(5)
    
    # Start GUI
    gui_process = start_gui()
    
    # Keep the main process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        
        # Terminate processes
        if server_process:
            server_process.terminate()
        if gui_process:
            gui_process.terminate()

if __name__ == "__main__":
    main()
